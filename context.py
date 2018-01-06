import multiprocessing
import time
from myqueue import MyQueue

class Context:
    def __init__(self):
        self.profiler = None
        self.keepalive = multiprocessing.Value('d', time.time())
        self.remotes = {}