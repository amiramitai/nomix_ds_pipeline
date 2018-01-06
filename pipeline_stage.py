import multiprocessing
import os
import tensorflow as tf
import numpy as np

from myqueue import MyQueue
from data import DataClass, DataType, SimpleDualDS

from vgg16 import Vgg16
from audio import get_image_with_audio, to_audiosegment

from exceptions import NoThreadSlots


class PipelineStage:
    def __init__(self, config, context):
        self._input_queue = None
        self._name = config.get('name')
        self._params = config.get('params')
        self._output_queue = MyQueue(config, context)
        self._context = context
        self._threads_alloc = multiprocessing.Value('i', 0)
        self._threads_occup = multiprocessing.Value('i', 0)
        self._thread_alloc_lock = multiprocessing.Lock()
        self.index = 0
        self._is_last = False
        self._max_parallel = multiprocessing.cpu_count()
        self.should_display = True
        self._cache_config = None

    def in_proc_init(self):
        pass

    def get_max_parallel(self):
        return self._max_parallel

    def use_thread_slot(self):
        with self._thread_alloc_lock:
            if self._threads_alloc.value < 1:
                raise NoThreadSlots(self.name)
            
            self._threads_occup.value += 1
            self._threads_alloc.value -= 1

    def free_thread_slot(self):
        with self._thread_alloc_lock:
            if self._threads_occup.value == 0:
                print('[!!!!] WTF?!?!')
            self._threads_occup.value -= 1

    def get_occupied_threads(self):
        return self._threads_occup.value

    def get_thread_alloc(self):
        with self._thread_alloc_lock:
            ret = self._threads_alloc.value
            return ret

    def set_thread_alloc(self, val):
        with self._thread_alloc_lock:
            if val < 0:
                RuntimeError('BUG: negative thread alloc', self.name, val)
            self._threads_alloc.value = val

    @property
    def output_queue(self):
        return self._output_queue

    @property
    def input_queue(self):
        return self._input_queue

    @input_queue.setter
    def input_queue(self, _input):
        self._input_queue = _input
        return self._input_queue

    @property
    def name(self):
        if self._name:
            return self._name
        self._name_cache = self.__class__.__name__
        if self._name_cache.endswith('Stage'):
            self._name_cache = self._name_cache[:-5]
        return self._name_cache

    def write(self, data):
        raise NotImplementedError()


class DatasetStage(PipelineStage):
    def __init__(self, config, context, dataset):
        super().__init__(config, context)
        self._ds = dataset

    def write(self, data):
        if not isinstance(data, int):
            raise RuntimeError('Expected an integer')
        # print('[+] getting {} vars'.format(data))

        gen = self._ds.read(data)
        for item in gen:
            # print('[+] {}::write put'.format(self.__class__.__name__))
            # print(item)
            self.output_queue.put(item)


class DualDatasetStage(DatasetStage):
    def __init__(self, config, context):
        super().__init__(config, context, SimpleDualDS(config['params']))


# class AudioMixerStage(PipelineStage):
#     def write(self, data):
#         vocls, insts = data
#         comb = []
#         # import pdb; pdb.set_trace()
#         pool = multiprocessing.Pool()
#         comb = pool.imap(self._combine, enumerate(zip(vocls, insts)))
#         return comb

#     def _combine(self, args):
#         i, (y1, y2) = args
#         vocl_seg = to_audiosegment(y1)
#         inst_seg = to_audiosegment(y2)

#         combined = np.array(inst_seg.overlay(vocl_seg).get_array_of_samples())
#         return combined, 1


class AudioJoinStage(PipelineStage):
    def write(self, data):
        for vocl, inst in data:
            self.output_queue.put((vocl, 1))
            self.output_queue.put((inst, 0))


class AudioToImageStage(PipelineStage):
    def write(self, data):
        image = get_image_with_audio(*data)
        self.output_queue.put(image)


class ImageToEncodingStage(PipelineStage):
    def __init__(self, config, context):
        super().__init__(config, context)
        self._max_parallel = 1

    def in_proc_init(self):
        self.sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        weights = self._params.get('weights', 'vgg16_weights.npz')
        self.vgg = Vgg16(imgs,
                         weights,
                         self.sess,
                         weights_to_load_hack=28)
    
    def write(self, data):
        img, label = data
        img = np.repeat(img.reshape(224, 224, 1), 3, axis=2)
        embed = self.sess.run(self.vgg.fc1,
                                feed_dict={self.vgg.imgs: [img]})[0]
        self.output_queue.put((embed, label))


class PrinterStage(PipelineStage):
    def __init__(self, config, context):
        super().__init__(config, context)
        self.counter = 0

    def write(self, data):
        self.counter += 1
        print('[+] {}. PrinterStage: {}'.format(self.counter, str(data)[:40] + '...'))

class PrintSummary(PipelineStage):
    def __init__(self, config, context):
        super().__init__(config, context)
        self.finished = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        self.should_display = False

    def write(self, data):
        with self.lock:
            self.finished.value += 1

        print('[+] PrintSummary: {}. {} {}'.format(self.finished.value, data[0].shape, data[1]))
