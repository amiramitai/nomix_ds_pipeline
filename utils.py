import threading
import collections
import queue as Queue
import multiprocessing
import os


def iterate_audio_files(path):
    return iterate_files(path, ('.wav', '.aac', '.wav', '.ogg', '.wma', '.m4a'))

def iterate_files(path, ext=''):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                yield os.path.join(root, file)


class StringValue:
    def __init__(self, msg=None):
        self._val = multiprocessing.Array('c', 30)
        if msg:
            self.value = msg

    @property
    def value(self):
        return self._val.value.decode('utf-8')

    @value.setter
    def value(self, val):
        self._val.value = val.encode('utf-8')
        return val


class AsyncGenerator:
    """
    The AsyncGenerator class is used to buffer output of a
    generator between iterable.__next__ or iterable.next calls. This
    allows the generator to continue producing output even if the
    previous output has not yet been consumed. The buffered structure is
    particularly useful when a process that consumes data from a
    generator is unable to finish its task at a rate comparable to which
    data is produced such as writing a large amount of data to a
    low-bandwidth I/O stream at the same time the data is produced.

        >>> for chunk in AsyncGenerator(function=makes_lots_of_data):
        ...     really_slow_iostream.write(chunk)
    """
    def __init__(self, generator, start=True, maxsize=0):
        # self.generator = iter(function(*args, **kwargs))
        self.generator = generator
        self.thread = threading.Thread(target=self._generatorcall)
        self.q = Queue.Queue(maxsize=maxsize)
        self.next = self.__next__
        if start:
            self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        done, item = self.q.get()
        if done:
            raise StopIteration
        else:
            return item

    def _generatorcall(self):
        try:
            for output in self.generator:
                self.q.put((False, output))
        finally:
            self.q.put((True, None))