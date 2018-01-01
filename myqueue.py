import multiprocessing
import os
import uuid
from cache_file import CacheCollection, StopCache


QUEUE_MAX_SIZE = 64
EMPTY_QUEUE_SLEEP = 0.1
WHITE_BG = '\033[5;30;47m'
GREEN_BG = '\033[0;30;42m'
BLUE_BG = '\033[0;30;44m'
RED_BG = '\033[0;30;41m'
GREEN = '\033[1;32;40m'
NC = '\033[0m'
PROGRESSBAR_STAGES = 15

class BaseQueue:
    def get_input_info(self):
        input_info = list(('{:%dd}'%(PROGRESSBAR_STAGES)).format(self.qsize()))
        fract = min(self.qsize(), QUEUE_MAX_SIZE) / QUEUE_MAX_SIZE
        level = int(fract * PROGRESSBAR_STAGES)
         
        cur = 0
        if self.is_caching():
            self.refresh_cache()
            cur = self.get_approx_cursor()
        cur_level = int(cur * (PROGRESSBAR_STAGES-1))

        # colors
        bg_color = WHITE_BG
        if self.is_caching():
            bg_color = GREEN_BG
        fg_color = NC
        if self.is_caching():
            fg_color = GREEN
        restore_color = bg_color
        if cur > fract:
            restore_color = fg_color

        if self.is_caching():
            input_info[cur_level-1] = ''.join([RED_BG, input_info[cur_level-1], restore_color])
        input_info[0] = ''.join([bg_color, input_info[0]])
        input_info[level-1] = ''.join([fg_color, input_info[level-1]])
        input_info[-1] = ''.join([input_info[-1], NC])
        
        return ''.join(input_info)

    def is_caching(self):
        return False

    def get_approx_cursor(self):
        raise RuntimeError('Should not be called')

class MyQueue(BaseQueue):
    def __init__(self, maxsize=None, last=False):
        self.size = multiprocessing.Value('i', 0)
        self.counter = multiprocessing.Value('i', 0)
        self._is_caching = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        if maxsize:
            self.queue = multiprocessing.Queue(maxsize=maxsize)
        else:
            self.queue = None
        self.last = last
        self._cache = None

    def set_cache(self, cache_config):
        with self.lock:
            # print('[+] set_cache', cache_path)
            self._cache = CacheCollection(cache_config)
            self._is_caching.value = 1

    def get_approx_cursor(self):
        return self._cache.get_approx_cursor()

    def is_caching(self):
        return self._is_caching.value > 0

    def refresh_cache(self):
        if self.is_caching():
            self._cache._load_all()
    
    def _safe_write_cache(self, item):
        if not self._cache:
            return
        with self.lock:
            try:
                self._cache.write(item)
            except StopCache:
                print('[+] Got stop cache..')
                self._cache = None
                self._is_caching.value = 0

    def put(self, item):
        self._safe_write_cache(item)
        if not self.last:
            ret = self.queue.put(item)
        with self.lock:
            self.size.value += 1
            self.counter.value += 1


    def get(self, *args, **kwargs):
        if self.last:
            raise RuntimeError('Nobody should be reading this')
        ret = self.queue.get(*args, **kwargs)
        with self.lock:
            self.size.value -= 1
        return ret

    def qsize(self):
        return self.size.value

class MyInputQueue(BaseQueue):
    def __init__(self):
        self.samples_to_read = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        self._cache = None

    def put(self, samples):
        pass

    def get(self, *args, **kwargs):
        return 2

    def qsize(self):
        return QUEUE_MAX_SIZE