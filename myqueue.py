import multiprocessing
import os
import uuid
from cache_file import CacheCollection, StopCache


DEFAULT_QUEUE_CAPACITY = 128
EMPTY_QUEUE_SLEEP = 0.1
WHITE_BG = '\033[5;30;47m'
GREEN_BG = '\033[0;30;42m'
BLUE_BG = '\033[0;30;44m'
RED_BG = '\033[0;30;41m'
GREEN = '\033[1;32;40m'
NC = '\033[0m'
PROGRESSBAR_STAGES = 15
QUEUE_BLOCK_TIMEOUT = 0.5

class QueueOverflow(Exception):
    pass

class BaseQueue:
    def __init__(self, config=None):
        _config = config
        if not config:
            _config = {}
        
        self._max_size = _config.get('max_size', DEFAULT_QUEUE_CAPACITY)
        self.lock = multiprocessing.Lock()
        self._cache = None
        self._drop = False

    def get_capacity(self):
        return self._max_size

    def get_input_info(self):
        qsize = self.qsize()
        disp = self.get_display_amount()
        input_info = list(('{:%dd}'%(PROGRESSBAR_STAGES)).format(disp))
        fract = min(qsize, self._max_size) / self._max_size
        level = int(fract * (PROGRESSBAR_STAGES-1))
         
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
            input_info[cur_level] = ''.join([RED_BG, input_info[cur_level], restore_color])
        input_info[0] = ''.join([bg_color, input_info[0]])
        input_info[level] = ''.join([input_info[level], fg_color])
        input_info[-1] = ''.join([input_info[-1], NC])
        
        return ''.join(input_info)

    def is_caching(self):
        return False

    def get_approx_cursor(self):
        raise RuntimeError('Should not be called')

    def get_display_amount(self):
        return self.qsize()

class MyQueue(BaseQueue):
    def __init__(self, config):
        super().__init__(config)
        # drop items instead of queueing
        self._drop = config.get('drop', False)
        self.size = multiprocessing.Value('i', 0)
        self.counter = multiprocessing.Value('i', 0)
        self._is_caching = multiprocessing.Value('i', 0)
        self.queue = multiprocessing.Queue(self.get_capacity())
        if 'cache' in config:
            self._set_cache(config['cache'])

        self._config = config


    def _set_cache(self, cache_config):
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
        if not self._drop:
            # print('[+] {}::put'.format(self._config['name']))
            ret = self.queue.put(item, True, QUEUE_BLOCK_TIMEOUT)
        self._safe_write_cache(item)
        with self.lock:
            self.size.value += 1
            self.counter.value += 1


    def get(self, *args, **kwargs):
        if self._drop:
            raise RuntimeError('dropping queues should not be read')
        ret = self.queue.get(*args, **kwargs)
        with self.lock:
            self.size.value -= 1
        return ret

    def qsize(self):
        if self._drop:
            return 0
        return self.size.value

    def get_display_amount(self):
        if self._drop:
            return self.counter.value
        return self.qsize()

class MyInputQueue(BaseQueue):
    def __init__(self):
        super().__init__()

    def put(self, samples):
        pass

    def get(self, *args, **kwargs):
        return 2

    def qsize(self):
        return self._max_size