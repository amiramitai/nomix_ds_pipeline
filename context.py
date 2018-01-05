import multiprocessing
import time

class Context:
    def __init__(self):
        self.profiler = None
        self.keepalive = multiprocessing.Value('d', time.time())