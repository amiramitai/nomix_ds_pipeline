import itertools
import multiprocessing
import numpy as np
import tensorflow as tf
import queue
import threading
import traceback
import time

from audio import get_image_with_audio, SAMPLE_RATE
from data import DataClass, DataType, SimpleDualDS
from pydub import AudioSegment
from utils import AsyncGenerator
from vgg16 import Vgg16


class PipelineStage:
    def __init__(self):
        self._input_queue = None
        self._output_queue = multiprocessing.Queue(maxsize=QUEUE_MAX_SIZE)

    def in_proc_init(self):
        pass

    @property
    def output_queue(self):
        return self._output_queue

    @output_queue.setter
    def output_queue(self, _output):
        self._output_queue = _output_queue
        return self._output_queue

    @property
    def input_queue(self):
        return self._input_queue

    @input_queue.setter
    def input_queue(self, _input):
        self._input_queue = _input
        return self._input_queue
    
    # def get_input_queue(self):
    #     assert 


    def write(self, data):
        raise NotImplementedError()


class DatasetStage(PipelineStage):
    def __init__(self, dataset):
        super().__init__()
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

def is_main_thread_alive():
        return any([a.is_alive() for a in threading.enumerate() if a.name == 'MainThread'])

def keepalive_worker(keepalive):
    # print('[+] in keepalive_worker')
    start = time.time()
    while is_main_thread_alive():
        keepalive.value = time.time()
        time.sleep(2.0)
        print('[+] time:', time.time() - start)
    # print('[+] keepalive_worker done1')

STAGE_GET_TIMEOUT = 5000
KEEP_ALIVE_TIMEOUT = 10000
def pipeline_stage_worker(stage, keepalive):
    print('[+] pipeline_stage_worker:', stage.__class__.__name__)
    stage.in_proc_init()
    while True:
        try:
            # print(stage)
            item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
            # print('\t[+] {} received an item!'.format(stage.__class__.__name__))
            stage.write(item)
            # print(out)
            # stage.output_queue.put(out)
        except queue.Empty:
            if (time.time() - keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout.. killing..')
                return
        except Exception as e:
            traceback.print_exc()


QUEUE_MAX_SIZE = 64
class Pipeline:
    def __init__(self):
        self._stages = []
        self._input_queue = multiprocessing.Queue(maxsize=QUEUE_MAX_SIZE)
        self._keepalive = multiprocessing.Value('d', time.time())
        self._keepalive_thread = None

    def get_samples(self, dtype, num):
        raise NotImplementedError()

    def _get_tip_queue(self):
        if not self._stages:
            return self._input_queue
        
        return self._stages[-1].output_queue

    def add_stage(self, stage):
        queue = self._get_tip_queue()
        stage.input_queue = queue
        self._stages.append(stage)


    def run(self):
        self._keepalive_thread = threading.Thread(target=keepalive_worker,
                                                  args=(self._keepalive,))
        self._keepalive_thread.start()
        
        ps = []
        for s in self._stages:
            print('[+] intializing process:', s.__class__.__name__)
            p = multiprocessing.Process(target=pipeline_stage_worker, args=(s, self._keepalive))
            ps.append(p)
            # p.daemon = True
        print('[+] starting processes')
        [p.start() for p in ps]
        

    def read(self, size):
        self._input_queue.put(size)
        tip = self._get_tip_queue()
        return [tip.get() for a in range(size)]
        # last_stage_data = size
        # all_before = time.time()
        # for s in self._stages:
        #     before = time.time()
        #     # import pdb; pdb.set_trace()
        #     last_stage_data = s.write(last_stage_data)
        #     delta = time.time() - before
        #     name = s.__class__.__name__
        #     print('\t[+] {} took: {}'.format(name, delta))
        # print('[+] all took: {}'.format(time.time() - all_before))
        # return last_stage_data

    def iterate(self, size):
        while True:
            ret = self.read(size)
            yield ret


class AudioMixerStage(PipelineStage):
    def write(self, data):
        vocls, insts = data
        comb = []
        # import pdb; pdb.set_trace()
        pool = multiprocessing.Pool()
        comb = pool.imap(self._combine, enumerate(zip(vocls, insts)))
        return comb

    def _combine(self, args):
        i, (y1, y2) = args
        # print('[+] {} combining...'.format(i))
        vocl_seg = self._to_audiosegment(y1)
        inst_seg = self._to_audiosegment(y2)

        combined = np.array(inst_seg.overlay(vocl_seg).get_array_of_samples())
        return combined, 1

    def _to_audiosegment(self, arr):
        if arr.dtype in [np.float16, np.float32, np.float64]:
            arr = np.int16(arr/np.max(np.abs(arr)) * 32767)
        
        return AudioSegment(arr.tobytes(),
                            frame_rate=SAMPLE_RATE,
                            sample_width=2,
                            channels=1)


class AudioJoinStage(PipelineStage):
    def write(self, data):
        for vocl, inst in data:
            # print('[+] {}::write put'.format(self.__class__.__name__))
            self.output_queue.put((vocl, 1))
            self.output_queue.put((inst, 0))


class AudioToImageStage(PipelineStage):
    def write(self, data):
        # pool = multiprocessing.Pool()
        # return pool.imap(get_image_with_audio, enumerate(data))
        # for aud in data:
        # print('[+] {}::write put'.format(self.__class__.__name__))
        # print(data)
        self.output_queue.put(get_image_with_audio(*data))



class ImageToEncoding(PipelineStage):
    def __init__(self):
        super().__init__()

    def in_proc_init(self):
        super().in_proc_init()
        self.sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = Vgg16(imgs,
                         'vgg16_weights.npz',
                         self.sess,
                         weights_to_load_hack=28)
    
    def write(self, data):
        # print(data)
        img, label = data
        img = np.repeat(img.reshape(224, 224, 1), 3, axis=2)
        embed = self.sess.run(self.vgg.fc1,
                                feed_dict={self.vgg.imgs: [img]})[0]
        # print('[+] {}::write put'.format(self.__class__.__name__))
        self.output_queue.put((embed, label))


class PrinterStage(PipelineStage):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def write(self, data):
        self.counter += 1
        print('[+] {}. PrinterStage: {}'.format(self.counter, str(data)[:40] + '...'))


class AudioMixerPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self._stages.append(DatasetStage(SimpleDualDS()))
        self._stages.append(AudioMixerStage())
        self._stages.append(AudioToImageStage())
        self._stages.append(ImageToEncoding())
        self._stages.append(PrinterStage())

    def read(self, size):
        return super().read(size*2)


class AudioEncoderPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.add_stage(DatasetStage(SimpleDualDS()))
        # self.add_stage(AudioJoinStage())
        self.add_stage(AudioToImageStage())
        self.add_stage(ImageToEncoding())
        # self.add_stage(TrainFormatterStage())
        self.add_stage(PrinterStage())


# class TrainFormatterStage(PipelineStage):
#     def write(self, data):
#         encs = []
#         labels = []
#         env, label = data
#         encs.append(enc)
#         hot_shot = [0, 0]
#         hot_shot[label] = 1
#         labels.append(hot_shot)
#         output.
#         return np.array(encs), np.array(labels)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        read_size = 8
        print('[+] Using default read_size..')
    else:
        read_size = int(sys.argv[1])
        print('[+] Using given read_size:', read_size)
    amp = AudioEncoderPipeline()
    amp.run()
    ret = amp.read(read_size)
    # import pdb; pdb.set_trace()
