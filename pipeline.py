from data import DataClass, DataType, SimpleDualDS
from pydub import AudioSegment
import numpy as np
import threading
import multiprocessing
import tensorflow as tf
from vgg16 import Vgg16
from audio import get_image_with_audio, SAMPLE_RATE
import time
from utils import AsyncGenerator
import itertools


class PipelineStage:

    def write(self, data):
        raise NotImplementedError()


class DatasetStage(PipelineStage):
    def __init__(self, dataset):
        self._ds = dataset

    def write(self, data):
        if not isinstance(data, int):
            raise RuntimeError('Expected an integer')
        # print('[+] getting {} vars'.format(data))
        return self._ds.read(data)


class Pipeline:
    def __init__(self):
        super().__init__()
        self._stages = []

    def get_samples(self, dtype, num):
        raise NotImplementedError()

    def read(self, size):
        last_stage_data = size
        all_before = time.time()
        for s in self._stages:
            before = time.time()
            # import pdb; pdb.set_trace()
            last_stage_data = s.write(last_stage_data)
            delta = time.time() - before
            name = s.__class__.__name__
            print('\t[+] {} took: {}'.format(name, delta))
        print('[+] all took: {}'.format(time.time() - all_before))
        return last_stage_data

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
        vocls, insts = data
        g1 = map(self._voc, vocls)
        g2 = map(self._inst, insts)
        return itertools.chain(g1, g2)

    def _voc(self, x):
        return x, 1

    def _inst(self, x):
        return x, 0


class AudioToImageStage(PipelineStage):
    def write(self, data):
        pool = multiprocessing.Pool()
        return pool.imap(get_image_with_audio, enumerate(data))


class ImageToEncoding(PipelineStage):
    def __init__(self):
        self.sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = Vgg16(imgs,
                         'vgg16_weights.npz',
                         self.sess,
                         weights_to_load_hack=28)
    
    def write(self, data):
        embeds = []
        for i, (img, label) in enumerate(AsyncGenerator(data)):
            # print('[+] {} embedding..'.format(i))
            img = np.repeat(img.reshape(224, 224, 1), 3, axis=2)
            embed = self.sess.run(self.vgg.fc1,
                                  feed_dict={self.vgg.imgs: [img]})[0]
            embeds.append((embed, label))
        return embeds


class PrinterStage(PipelineStage):
    def write(self, data):
        print('[+] PrinterStage:', len(data))
        return data


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
        self._stages.append(DatasetStage(SimpleDualDS()))
        self._stages.append(AudioJoinStage())
        self._stages.append(AudioToImageStage())
        self._stages.append(ImageToEncoding())
        self._stages.append(TrainFormatterStage())
        # self._stages.append(PrinterStage())


class TrainFormatterStage(PipelineStage):
    def write(self, data):
        encs = []
        labels = []
        for enc, label in data:
            encs.append(enc)
            hot_shot = [0, 0]
            hot_shot[label] = 1
            labels.append(hot_shot)
        return np.array(encs), np.array(labels)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        read_size = 8
        print('[+] Using default read_size..')
    else:
        read_size = int(sys.argv[1])
        print('[+] Using given read_size:', read_size)
    amp = AudioEncoderPipeline()
    ret = amp.read(read_size)
    # import pdb; pdb.set_trace()
