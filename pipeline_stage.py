import random
import multiprocessing
import os
import numpy as np
import queue

from myqueue import MyQueue
from data import NomixDS, MultiDatasets, MixWithVocalResult, InstWithVocalResult, FCN_VOC_THRESHOLD

from audio import get_image_with_audio, to_audiosegment, get_rand_audio_patch, \
                  get_mel_filename, get_offset_range_patch

from exceptions import NoThreadSlots
import traceback


class PipelineStage:
    def __init__(self, config, context):
        self._input_queue = None
        self._name = config.get('name')
        self._params = config.get('params')
        self._output_queue = MyQueue(config, context)
        self._output_queue2 = MyQueue(config, context)
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
            # self._threads_alloc.value -= 1

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
    def output_queue2(self):
        return self._output_queue2

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
        self._vocls = None
        self._insts = None

    def in_proc_init(self):
        self._vocls = self._ds.vocals()
        self._insts = self._ds.instrumentals()

    def write(self, data):
        if not isinstance(data, int):
            raise RuntimeError('Expected an integer')

        # print('[+] v = next(self._vocls)')
        v = next(self._vocls)
        if not isinstance(v, MixWithVocalResult):
            i = next(self._insts)    
            v = InstWithVocalResult(v, i)
        # print('[+] i = next(self._insts)')
        i = next(self._insts)
        self.output_queue.put(v)
        self.output_queue.put(i)


class MultiDatasetsStage(DatasetStage):
    def __init__(self, config, context):
        super().__init__(config, context, MultiDatasets(config['params']))
        self._max_parallel = 1

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

class HasNanException(Exception):
    pass


class FilenameToSlice(PipelineStage):
    def __init__(self, config, context):
        super().__init__(config, context)
        self._max_parallel = 6
    
    def write(self, data):
        try:
            with np.errstate(all='raise'):
                self._safe_write(data)
        except queue.Full:
            raise
        except HasNanException:
            print("[!] Nan detected, skipping..")
            open("db_nans.log", 'ab').write(str(self._get_desc(data)).encode('utf-8'))
            open("db_nans.log", 'a').write('------------------------------------')
        except Exception:
            print("[!] Warning detected, skipping..")
            tb = traceback.format_exc()
            open("numpy_warns.log", 'ab').write(str(self._get_desc(data)).encode('utf-8'))
            open("numpy_warns.log", 'ab').write(str(tb).encode('utf-8'))
            open("numpy_warns.log", 'a').write('------------------------------------')

    def _get_desc(self, data):
        if isinstance(data, (MixWithVocalResult, InstWithVocalResult)):
            return data.desc()

        params, label = data
        filename, offset, _range = params
        return filename

    def _safe_write(self, data):
        # print(data)
        if isinstance(data, MixWithVocalResult):
            self.output_queue.put((data.desc(), data._slice(), data.get_label()))
            return
        if isinstance(data, InstWithVocalResult):
            self.output_queue.put((data.desc(), data.mix(), data.get_label()))
            return
        
        params, label = data
        filename, offset, _range = params

        aud = get_offset_range_patch(filename, offset, _range)

        # if np.isnan(aud).any() or np.isinf(aud).any():
        if np.isnan(aud).any():
            raise HasNanException()

        desc = str(params)

        self.output_queue.put((desc, aud, label))


class FilenameToSliceFcn(FilenameToSlice):
    def _safe_write(self, data):
        # print(data)
        if isinstance(data, MixWithVocalResult):
            self.output_queue.put((data.desc(), data._slice(), data.get_fcn_label()))
            return
        if isinstance(data, InstWithVocalResult):
            self.output_queue.put((data.desc(), data.mix(), data.get_fcn_label()))
            return
        
        params, label = data
        filename, offset, _range = params


        aud = get_offset_range_patch(filename, offset, _range)
        
        if label == [1, 0]:
            voc = np.zeros((aud.shape[0], aud.shape[1], 1))
            others = (aud > FCN_VOC_THRESHOLD).astype('float')
            others = others.reshape((aud.shape[0], aud.shape[1], 1))
            label = np.concatenate((others, voc), axis=2) 
        else:
            voc = (aud > FCN_VOC_THRESHOLD).astype('float')
            others = np.zeros(voc.shape)
            voc = voc.reshape((voc.shape[0], voc.shape[1], 1))
            others = others.reshape((others.shape[0], others.shape[1], 1))
            label = np.concatenate((others, voc), axis=2)

        # if np.isnan(aud).any() or np.isinf(aud).any():
        if np.isnan(aud).any():
            raise HasNanException()

        desc = str(params)

        self.output_queue.put((desc, aud, label))


class FilenameToSliceFrrn(FilenameToSlice):
    def _safe_write(self, data):
        # print(data)
        if isinstance(data, MixWithVocalResult):
            self.output_queue.put((data.desc(), data._slice(), data.get_frrn_label()))
            return
        if isinstance(data, InstWithVocalResult):
            self.output_queue.put((data.desc(), data.mix(), data.get_frrn_label()))
            return
        
        params, label = data
        filename, offset, _range = params

        aud = get_offset_range_patch(filename, offset, _range)
        
        if label == [1, 0]:
            voc = np.zeros((aud.shape[0], aud.shape[1], 1))
            # others = (aud > FCN_VOC_THRESHOLD).astype('float')
            # others = others.reshape((aud.shape[0], aud.shape[1], 1))
            voc = (voc > FCN_VOC_THRESHOLD).astype('int32')
            label = voc
            # label = np.concatenate((others, voc), axis=2) 
        else:
            # voc = (aud > FCN_VOC_THRESHOLD).astype('float')
            voc = aud
            voc = voc.reshape((voc.shape[0], voc.shape[1], 1))
            voc = (voc > FCN_VOC_THRESHOLD).astype('int32')
            label = voc

        # if np.isnan(aud).any() or np.isinf(aud).any():
        if np.isnan(aud).any():
            raise HasNanException()

        desc = str(params)

        self.output_queue.put((desc, aud, label))

class FilenameToSliceFrrn2(FilenameToSlice):
    def _safe_write(self, data):
        # print(data)
        if isinstance(data, MixWithVocalResult):
            self.output_queue.put((data.desc(), data._slice(), data.get_frrn2_label()))
            return
        if isinstance(data, InstWithVocalResult):
            self.output_queue.put((data.desc(), data.mix(), data.get_frrn2_label()))
            return
        
        params, label = data
        filename, offset, _range = params

        aud = get_offset_range_patch(filename, offset, _range)
        
        if label == [1, 0]:
            others = aud.reshape((aud.shape[0], aud.shape[1], 1))
            voc = np.zeros((aud.shape[0], aud.shape[1], 1))
            label = np.concatenate((others, voc), axis=2) 
        else:
            # voc = (aud > FCN_VOC_THRESHOLD).astype('float')
            others = np.zeros((aud.shape[0], aud.shape[1], 1))
            voc = aud.reshape((voc.shape[0], voc.shape[1], 1))
            label = np.concatenate((others, voc), axis=2) 
            # voc = (voc > FCN_VOC_THRESHOLD).astype('int32')
            # label = voc

        # if np.isnan(aud).any() or np.isinf(aud).any():
        if np.isnan(aud).any():
            raise HasNanException()

        desc = str(params)

        self.output_queue.put((desc, aud, label))


class FilenameToSliceRnn(FilenameToSlice):
    def _safe_write(self, data):
        # print(data)
        if isinstance(data, MixWithVocalResult):
            self.output_queue.put((data.desc(), data._slice(), *data.get_rnn_label()))
            return
        if isinstance(data, InstWithVocalResult):
            self.output_queue.put((data.desc(), data.mix(), *data.get_rnn_label()))
            return
        
        params, label = data
        filename, offset, _range = params

        aud = get_offset_range_patch(filename, offset, _range)
        
        if label == [1, 0]:
            y1 = aud.reshape((aud.shape[0], aud.shape[1], 1))
            y2 = np.zeros((aud.shape[0], aud.shape[1], 1))
            # label = np.concatenate((others, voc), axis=2) 
        else:
            # voc = (aud > FCN_VOC_THRESHOLD).astype('float')
            y1 = np.zeros((aud.shape[0], aud.shape[1], 1))
            y2 = aud.reshape((voc.shape[0], voc.shape[1], 1))
            # label = np.concatenate((others, voc), axis=2) 
            # voc = (voc > FCN_VOC_THRESHOLD).astype('int32')
            # label = voc

        # if np.isnan(aud).any() or np.isinf(aud).any():
        if np.isnan(aud).any():
            raise HasNanException()

        desc = str(params)

        self.output_queue.put((desc, aud, y1, y2))
        # self.output_queue2.put((desc, aud, y1, y2))


class FilenameToAudio(PipelineStage):
    def write(self, data):
        _range = None
        filename, label = data
        if isinstance(filename, tuple):
            filename, _range = filename

        aud = get_rand_audio_patch(filename, _range)
        self.output_queue.put((aud, label))
    

class AudioToImageStage(PipelineStage):
    def write(self, data):
        image = get_image_with_audio(*data)
        self.output_queue.put(image)


class ImageToEncodingStage(PipelineStage):
    def __init__(self, config, context):
        super().__init__(config, context)
        self._max_parallel = 1

    def in_proc_init(self):
        import tensorflow as tf
        from vgg16 import Vgg16
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
