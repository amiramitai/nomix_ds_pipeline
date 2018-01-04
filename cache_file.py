import struct
import pickle
import glob
import os
import multiprocessing
import random

PADDING_SIZE = 64
SAMPLES_TO_FLUSH = 20
DEFAULT_MAX_SPLIT = 1
MAX_CACHEFILES_DIGITS = 4 

class CacheSeekPolicy:
    ONE_SHOT = 0
    CYCLIC = 1
    RANDOM = 2

class CacheMode:
    READ = 0
    WRITE = 1

class StopCache(Exception):
    pass

class EofException(Exception):
    pass

class CacheCollection:
    def __init__(self, config):
        self._config = config
        self._filename = config['filename']
        self._max_split = config.get('max_split', DEFAULT_MAX_SPLIT)
        self._max_records = config.get('max_records')
        self._max_size = config.get('max_size')
        if not self._max_records and not self._max_size and self._max_split > 1:
            print('[!] has split but no limit to recordsnum/filesize')
            self._max_split = 1
        self._caches = []
        self._loaded_caches = []
        self._fcursor = multiprocessing.Value('i', 0)
        self._stop_cache = multiprocessing.Value('b', 0)
        self._last_created_cache = 0
        self._lock = multiprocessing.RLock()
        self._seekpolicy = getattr(CacheSeekPolicy, config.get('seek_policy', 'CYCLIC'))
        self._not_yet_full = []
        self._load_all()

    def get_approx_cursor(self):
        if not self._caches:
            return 0
        
        if self._max_records:
            whole = self._max_records * self._max_split
            fcurs = self._fcursor.value
            cache_tell = 0
            if fcurs < len(self._caches):
                cache_tell = self._caches[fcurs].tell()
            cur = fcurs * self._max_records + cache_tell
            return cur / whole
        elif self._max_size:
            h = self._caches[0]._header
            sample_size = h.sample_size
            whole = self._max_size * self._max_split
            fcurs = self._fcursor.value
            cache_tell = 0
            if fcurs < len(self._caches):
                cache_tell = h.size() + self._caches[fcurs].tell() * sample_size
            cur = fcurs * self._max_size + cache_tell
            return cur / whole

        return 0       

    def read(self):
        # print('[+] CacheCollection::read', self._filename)
        with self._lock:
            self._load_all()
            i = self._fcursor.value
            while i < len(self._caches):
                cur_cache = self._caches[i]
                item = cur_cache.read()
                if item:
                    return item

    def write(self, item):
        with self._lock:
            if self._stop_cache.value > 0:
                return
            self._load_all()
            # print('[+] CacheCollection::write fcursor', self._fcursor.value, self._filename, os.getpid())
            i = self._fcursor.value
            while i < self._max_split:
                # print('[+] CacheCollection::write', i, self._filename)
                if i >= len(self._caches):
                    # print('[+] i={} len(self._caches)={}'.format(i, len(self._caches)))
                    self._create_next()
                
                cur_cache = self._caches[i]
                self._seek_with_policy(cur_cache, i)
                self._fcursor.value = i
                if self._requires_switching(cur_cache, CacheMode.WRITE):
                    i, reloop = self._iterate_with_policy(i, CacheMode.WRITE)
                    if reloop:
                        continue
                    cur_cache = self._caches[i]
                    self._fcursor.value = i
                # print('[+] writing..', i, cur_cache.tell())
                cur_cache.write(item)
                return
            self._stop_cache.value = 1
            raise StopCache('Max cache files split')

    def _iterate_with_policy(self, i, mode):
        if self._seekpolicy == CacheSeekPolicy.RANDOM:
            if mode == CacheMode.WRITE and len(self._caches) < self._max_split:
                # print('[+] needing a new cache file split')
                return len(self._caches), True
            return random.randint(0, len(self._caches)-1), False

        if self._should_rewind(i, mode):
            # print('[+] write - rewinding', self._filename, i)
            return 0, True

        return i + 1, True

    def _seek_with_policy(self, cache, i):
        if not cache.is_full() or self._not_yet_full:
            cache.end()
            return
        
        if self._fcursor.value != i:
            # if it's a different file then before
            if self._seekpolicy == CacheSeekPolicy.CYCLIC:
                cache.seek(0, 0)
                return
        
        if self._seekpolicy == CacheSeekPolicy.RANDOM:
            cache.seek(random.randint(0, cache.get_num_records()), 0)

    def _should_rewind(self, i, mode):
        if self._seekpolicy != CacheSeekPolicy.CYCLIC:
            return False
        # print('[+] _should_rewind:', i, MAX_CACHEFILES_SPLIT)
        if mode == CacheMode.WRITE and i < self._max_split - 1:
            return False

        if mode == CacheMode.READ and i < len(self._caches) - 1:
            return False

        return True

    def _requires_switching(self, cache, mode):
        is_full = cache.is_full()
        if is_full and self._not_yet_full:
            # fill them all up first
            self._not_yet_full = [c for c in self._caches if not c.is_full()]
            return True

        if self._seekpolicy == CacheSeekPolicy.RANDOM:
            if not is_full:
                return False
            return True

        if self._max_records and cache.tell() >= self._max_records:
            # print('[+] MAX_CACHEFILE_RECORDS', self._filename, cache.tell())
            return True

        if self._max_size and (cache._get_size() + cache._header.sample_size) > self._max_size:
            if cache.tell() >= cache.get_num_records():
                # print('[+] MAX_CACHEFILE_SIZE', self._filename, cache._get_size())
                return True

        return False

    def _load_all(self):
        # print('[+] CacheCollection::_load_all', self._filename)
        if len(self._caches) >= self._max_split:
            # No use to to try and reload
            return
        pattern = '{}.{}'.format(self._filename, '[0-9]' * MAX_CACHEFILES_DIGITS)
        cachelist = glob.glob(pattern)
        if len(cachelist) <= len(self._caches):
            # it doesn't seem to have new caches for us.. 
            return
        for c in cachelist:
            if c in self._loaded_caches:
                # print('[+] skipping a loaded cache:', c)
                continue
            # print('[+] found a new cache. loading:', c)
            config = dict(self._config)
            config['filename'] = c
            cf = CacheFile(config)
            cf.end()
            self._caches.append(cf)
            if not cf.is_full():
                self._not_yet_full.append(cf)
            self._loaded_caches.append(c)

    def _create_next(self):
        # print('[+] CacheCollection::_create_next', self._filename, len(self._caches))
        config = dict(self._config)
        config['filename'] = self._get_next_avail_name()
        self._caches.append(CacheFile(config))

    def _get_next_avail_name(self):
        # print('[+] CacheCollection::_get_next_avail_name', self._filename, self._last_created_cache)
        for i in range(self._last_created_cache, self._max_split):
            fmt = '{}.{:0%dd}' % MAX_CACHEFILES_DIGITS
            next_filename = fmt.format(self._filename, i)
            # print('[+] next_filename:', next_filename)
            if not os.path.isfile(next_filename):
                # print('[+] good filename:', next_filename)
                return next_filename
            # print('[+] finding another filename')
        self._stop_cache.value =1
        raise StopCache('max cache files split')

class CacheFileHeader:
    _fmt = '<L'
    def __init__(self):
        self.sample_size = 0

    def dumps(self):
        return struct.pack(self._fmt, self.sample_size)

    def loads(self, buff):
        self.sample_size, = struct.unpack(self._fmt, buff)

    def size(self):
        return struct.calcsize(self._fmt)


class CacheFile:
    def __init__(self, config):
        self._config = config
        self._filename = config['filename']
        self._max_records = config.get('max_records')
        self._max_size = config.get('max_size')
        self._header = CacheFileHeader()
        self._num_records = multiprocessing.Value('i', 0)
        # create file if does not exist
        self._initialized = False
        self._lock = multiprocessing.RLock()
        self._fcursor = multiprocessing.Value('i', 0)
        self._init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()
    
    def _init(self):
        # print('\t[+] _init', self._filename)
        if self._initialized:
            return
        with open(self._filename, 'ab'):
            pass
        # self._f = open(self._filename, 'r+b')
        # self._f.seek(0, 0)
        with self._lock, self._open() as f:
            buff = f.read(self._header.size())
            if buff:
                self._header.loads(buff)  
        self._initialized = True

    def __del__(self):
        self._close()

    def is_full(self):
        if self._max_size and (self._get_size() + self._header.sample_size) > self._max_size:
            return True

        if self._max_records and self.get_num_records() >= self._max_records:
            return True

        return False

    def _open(self):
        return open(self._filename, 'r+b')    
    
    def _close(self):
        # print('[+] _close', self._filename)
        if not self._initialized:
            return
        # self._f.close()    
        self._initialized = False

    def _set_sample_size(self, size):
        # print('\t[+] _set_sample_size', self._filename)
        if self._header.sample_size > 0:
            return
        size += PADDING_SIZE
        self._header.sample_size = size
        with self._lock, self._open() as f:
            # print('\t[+] _set_sample_size: rewriting header', self._filename)
            f.seek(0, 0)
            f.write(self._header.dumps())


    def _get_padded_buffer(self, buff):
        # print('\t[+] _get_padded_buffer', self._filename, self._header.sample_size, len(buff))
        to_pad = self._header.sample_size - len(buff)
        return buff + (chr(to_pad) * to_pad).encode('utf-8')


    def seek(self, i, mode=0):
        # print('\t[+] seek', i, mode, self._filename)
        with self._lock:
            if mode == 0:
                # abs_off = self._header.size() + self._header.sample_size * i
                self._fcursor.value = i
            elif mode == 1:
                self._fcursor.value += i
            elif mode == 2:
                self._fcursor.value = max(self._num_records.value - i, 0)
        

    def end(self):
        # print('\t[+] end', self._filename)
        self.seek(0, 2)

    def tell(self):
        # print('\t[+] tell', self._filename)
        return self._fcursor.value

    def get_num_records(self):
        if self._header.sample_size == 0:
            return 0
        
        with self._lock:
        # print('[+] get_records: recalculating self._num_records')
            datasize = self._get_size() - self._header.size()
            ret = int(datasize / self._header.sample_size)
            self._num_records.value = ret
            return ret

    def _get_size(self):
        with self._lock, self._open() as f:
            # print('\t[+] _get_size', self._filename)
            f.seek(0, 2)
            return f.tell()

    def read(self):
        # print('\t[+] read', self._filename)
        with self._lock, self._open() as f:
            f.seek(self._header.size() + self._fcursor * self._header.sample_size)
            buff = f.read(self._header.sample_size)
            if not buff:
                return None
            return pickle.loads(self._unwrap_buffer_pad(buff))

    def _unwrap_buffer_pad(self, padded):
        return padded[:-padded[-1]]

    def write(self, sample):
        # print('\t[+] write', self._filename)
        with self._lock, self._open() as f:
            # print('\t[+] write to', self._filename, self._fcursor.value)
            buff = pickle.dumps(sample)
            self._set_sample_size(len(buff))
            buff = self._get_padded_buffer(buff)
            f.seek(self._header.size() + self._header.sample_size * self._fcursor.value)
            f.write(buff)
            self._num_records.value += 1
            self._fcursor.value += 1
            f.flush()
            # if self._num_records % SAMPLES_TO_FLUSH == 0:
            #     print('flushing')
            #     self._f.flush()

if __name__ == '__main__':
    # print('[+] opening for write')
    print(getattr(CacheSeekPolicy, {}.get('policy', 'CYCLIC')))
    # with CacheFile('/tmp/test') as cf:
    #     cf.end()
    #     cf.write(1)
    #     cf.write(2)
    # print('[+] reopening for read')
    # with CacheFile('/tmp/test') as cf:
    #     print('[+] num_records:', cf.get_num_records())
    #     print(cf.read())
    #     print(cf.read())

    # with CacheFile('/Users/amiramitai/cache/AudioEncoderPipeline/0_DualDatasetStage') as cf:
    #     data = 1
    #     num_rec = cf.get_num_records()
    #     for i in range(num_rec):
    #         data = cf.read()
    #         if not data:
    #             break
    #         print('[{}/{}] {}: {}'.format(i+1, num_rec, str(data)[:30], data[1]))
    #         a = input()
    #         if a in ['q', 'Q']:
    #             break

    