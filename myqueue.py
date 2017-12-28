import multiprocessing
import os
import pickle
import uuid


QUEUE_MAX_SIZE = 64
EMPTY_QUEUE_SLEEP = 0.1


class MyQueue:
    def __init__(self, maxsize=None, last=False):
        self.size = multiprocessing.Value('i', 0)
        self.counter = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        if maxsize:
            self.queue = multiprocessing.Queue(maxsize=maxsize)
        else:
            self.queue = None
        self.last = last
        self._cache_folder = None

    def set_cache(self, folder):
        self._cache_folder = folder
        try:
            os.makedirs(folder)
        except:
            pass

    def put(self, item):
        if self._cache_folder:
            filepath = os.path.join(self._cache_folder, uuid.uuid4().hex)
            pickle.dump(item, open(filepath, 'wb'))
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

class MyInputQueue:
    def __init__(self):
        self.samples_to_read = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()

    def put(self, samples):
        pass


    def get(self, *args, **kwargs):
        return 2

    def qsize(self):
        return QUEUE_MAX_SIZE