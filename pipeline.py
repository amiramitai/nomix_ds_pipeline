import itertools
import multiprocessing
import numpy as np
import tensorflow as tf
import queue
import threading
import traceback
import time
import sys
import os

from audio import get_image_with_audio, SAMPLE_RATE
from data import DataClass, DataType, SimpleDualDS
from pydub import AudioSegment
from utils import AsyncGenerator
from vgg16 import Vgg16

KEEP_ALIVE_WORKER_SLEEP = 0.2
EMPTY_QUEUE_SLEEP = 0.1
QUEUE_MAX_SIZE = 64


def keyboard_int_guard(func):
    def _inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print('[!] Recieved KeyboardInterrupt.. exit(1)', os.getpid())
            sys.exit(1)

    return _inner


class MyQueue:
    def __init__(self, maxsize):
        self.size = multiprocessing.Value('i', 0)
        self.counter = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        self.queue = multiprocessing.Queue(maxsize=maxsize)

    def put(self, *args, **kwargs):
        ret = self.queue.put(*args, **kwargs)
        with self.lock:
            self.size.value += 1
            self.counter.value += 1
        return ret


    def get(self, *args, **kwargs):
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
        # print('[+] MyInputQueue::put', samples)
        # with self.lock:            
        #     self.samples_to_read.value += samples
        pass


    def get(self, *args, **kwargs):
        # print('[+] MyInputQueue::get')
        # while self.qsize() < 2:
        #     print('[+] MyInputQueue::get sleep')
        #     time.sleep(EMPTY_QUEUE_SLEEP)
        # with self.lock:
        #     self.samples_to_read.value -= 2
        return 2

    def qsize(self):
        # ret = min(self.samples_to_read.value / 2, QUEUE_MAX_SIZE)
        # ret = int(ret)
        # print('[+] MyInputQueue::qsize', ret)
        return QUEUE_MAX_SIZE

class PipelineStage:
    def __init__(self):
        self._input_queue = None
        self._output_queue = MyQueue(maxsize=QUEUE_MAX_SIZE)
        self._threads_alloc = multiprocessing.Value('i', 0)
        self._threads_occup = multiprocessing.Value('i', 0)
        self._thread_alloc_lock = multiprocessing.Lock()
        self.index = 0
        self._max_parallel = multiprocessing.cpu_count()
        self.should_display = True

    def in_proc_init(self):
        pass

    def get_max_parallel(self):
        return self._max_parallel

    def use_thread_slot(self):
        with self._thread_alloc_lock:
            if self._threads_alloc.value < 1:
                raise RuntimeError('BUG: No threads for allocation')
            
            self._threads_occup.value += 1
            self._threads_alloc.value -= 1

    def free_thread_slot(self):
        with self._thread_alloc_lock:
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
                print('[+] set_thread_alloc: {} BUG: {}'.format(self.name, val))
            self._threads_alloc.value = val

    def add_thread_alloc(self, alloc):
        with self._thread_alloc_lock:
            self._threads_alloc.value += alloc
            if self._threads_alloc.value < 0:
                print('[+] set_thread_alloc: {} BUG: {}/{}'.format(self.name, self._threads_alloc.value, alloc))

    @property
    def output_queue(self):
        return self._output_queue

    @output_queue.setter
    def output_queue(self, _output):
        self._output_queue = _output
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
        return self.__class__.__name__
    
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

PROGRESSBAR_STAGES = 10
def print_summary(start, stages):
    final_str = []
    time_passed = int(time.time() - start)
    total_occup_threads = 0
    for s in stages:
        input_info = ('{:%dd}'%(PROGRESSBAR_STAGES)).format(s.input_queue.qsize())
        fract = s.input_queue.qsize() / QUEUE_MAX_SIZE
        level = int(fract * PROGRESSBAR_STAGES)
        
        white = '\033[5;30;47m'
        nc = '\033[0m'
        formatted = [white]
        formatted.append(input_info[:level])
        formatted.append(nc)
        formatted.append(input_info[level:])
        input_info = ''.join(formatted)
        
        input_pipe = '[{}]'.format(input_info)
        if not s.should_display:
            final_str.append(input_pipe)
            continue
        occup_threads = s.get_occupied_threads()
        total_occup_threads += occup_threads
        alloc_threads = s.get_thread_alloc()
        # fract = s.input_queue.qsize() / QUEUE_MAX_SIZE
        # level = int(fract * PROGRESSBAR_STAGES)
        # amount_str = '[{}{}]'.format('#' * level, ' ' * (PROGRESSBAR_STAGES - level))
        # final_str.append('{}: {}[{}][{}]'.format(s.name, amount_str, alloc_threads, occup_threads))
        green = '\033[1;32;40m'
        yellow = '\033[1;33;40m'
        nc = '\033[0m'
        # state_color = green if occup_threads > 0 else nc
        state_color = nc
        if occup_threads > 0:
            state_color = green
        elif alloc_threads > 0:
            state_color = yellow
        pipe_rep = '{}{}[{}|{}]{}'.format(state_color, s.name, alloc_threads, occup_threads, nc)
        final_str.append('{}>>{}>>'.format(input_pipe, pipe_rep))
    final_str.append('[{}]'.format(s.output_queue.counter.value))
    final_str.append(' t[{}/{}] '.format(total_occup_threads, multiprocessing.cpu_count() ))
    final_str.append(str(time_passed))
    print(''.join(final_str), end='\r')


def is_main_thread_alive():
        return any([a.is_alive() for a in threading.enumerate() if a.name == 'MainThread'])


def prioritize_threads(stages):
    cpu_count = multiprocessing.cpu_count()
    used_threads = sum([s.get_occupied_threads() for s in stages])
    alloced_threads = sum([s.get_thread_alloc() for s in stages])
    avail_threads = cpu_count - (used_threads + alloced_threads)
    if avail_threads == 0:
        # print('[+] prioritize_threads: all threads were allocated.. returning')
        return
    # print('[+] prioritize_threads: cpu_count {}, used_threads {}, alloced_threads {}, avail_threads {}'\
        #   .format(cpu_count, used_threads, alloced_threads, avail_threads))
    if used_threads >= cpu_count:
        # print('[+] prioritize_threads: Lacking threads.. [{}/{}]'.format(avail_threads, cpu_count))
        return
    
    stages_to_prior = []
    priorities = []
    for s in stages:
        current_alloc = s.get_occupied_threads() + s.get_thread_alloc()
        if s.input_queue.qsize() == 0:
            continue

        if s.get_max_parallel() == 1:
            if current_alloc >= 1:
                continue
            s.add_thread_alloc(1)
            avail_threads -= 1
            continue
        
        input_priority = s.input_queue.qsize() / QUEUE_MAX_SIZE
        output_priority = 1.0 - (s.output_queue.qsize() / QUEUE_MAX_SIZE)
        current_alloc = s.get_occupied_threads() + s.get_thread_alloc()
        priority = input_priority * output_priority
        # if priority == 0:
        #     continue
        # print('[+] prioritize_threads: stage {} in {} out {}'.format(s.name, input_priority, output_priority))
        priorities.append(priority)
        stages_to_prior.append(s)

    if not stages_to_prior:
        return
    
    m = max(priorities)
    if m > 0:
        npriorities = []
        for p in priorities:
            npriorities.append((1.0/m) * p)
        priorities = npriorities
    
    zipped = zip(stages_to_prior, priorities)
    zipped = sorted(zipped, key=lambda k: k[1])
    zipped.reverse()
    for i in range(10):
        if avail_threads == 0:
            return
        for s, priority in zipped:
            current_alloc = s.get_occupied_threads() + s.get_thread_alloc()
            if current_alloc > s.get_max_parallel():
                continue
            t_to_alloc = int(priority * avail_threads)
            if t_to_alloc > s.get_max_parallel():
                # print('[+] prioritize_threads: fixing t_to_alloc > s.get_max_parallel(): {}->{}'.format(t_to_alloc, s.get_max_parallel()))
                t_to_alloc = max(s.get_max_parallel() - current_alloc, 0)
            
            t_to_alloc = min(t_to_alloc, avail_threads)
            # print('[+] prioritize_threads: stage {} gets {} threads'.format(s.name, t_to_alloc))
            s.add_thread_alloc(t_to_alloc)
            avail_threads -= t_to_alloc
    print('[+] prioritize_threads: could not allocate all threads: left -', avail_threads)

@keyboard_int_guard
def keepalive_worker(keepalive, stages):
    start = time.time()
    while is_main_thread_alive():
        keepalive.value = time.time()
        prioritize_threads(stages)
        print_summary(start, stages)
        time.sleep(KEEP_ALIVE_WORKER_SLEEP)

@keyboard_int_guard
def write_thread_occup_guard(start_event, thread_sem, stage, item):
    thread_sem.acquire()
    try:
        stage.use_thread_slot()
        if start_event:
            start_event.set()
        stage.write(item)
    finally:
        stage.free_thread_slot()
        thread_sem.release()


STAGE_GET_TIMEOUT = 5000
KEEP_ALIVE_TIMEOUT = 10000
@keyboard_int_guard
def no_fork_pipeline_stage_worker(thread_sem, stage, keepalive):
    # print('[+] pipeline_stage_worker({}): initializing..'.format(stage.name))
    stage.in_proc_init()
    while True:
        try:
            if stage.input_queue.qsize() == 0 or stage.get_thread_alloc() == 0:
                # print('[+] pipeline_stage_worker({}): no input.. sleeping..'.format(stage.name))
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            
            item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
            write_thread_occup_guard(None, thread_sem, stage, item)
        except queue.Empty:
            if (time.time() - keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout({}).. killing..'.format(stage.name))
                return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()


@keyboard_int_guard
def pipeline_stage_worker(thread_sem, stage, keepalive):
    # print('[+] pipeline_stage_worker({}): initializing..'.format(stage.name))
    stage.in_proc_init()
    spawned_procs = []
    while True:
        try:
            alive = []
            for p in spawned_procs:
                if p.is_alive():
                    alive.append(p)
                else:
                    del p
            spawned_procs = alive
            thread_alloc = stage.get_thread_alloc()
            # print(stage)
            # print('[+] pipeline_stage_worker({}): sizeof input: ({})'.format(stage.name, stage.input_queue.qsize()))
            if stage.input_queue.qsize() == 0 or thread_alloc == 0:
                # print('[+] pipeline_stage_worker({}): no input.. sleeping..'.format(stage.name))
                time.sleep(EMPTY_QUEUE_SLEEP)
                continue
            # print('[+] pipeline_stage_worker({}): working with {} threads'.format(stage.name, stage.get_thread_alloc()))
            for t in range(thread_alloc):
                item = stage.input_queue.get(True, STAGE_GET_TIMEOUT)
                # p = multiprocessing.Process(target=stage.write, args=(item,))
                proc_event = multiprocessing.Event()
                p = multiprocessing.Process(target=write_thread_occup_guard, args=(proc_event, thread_sem, stage, item))
                p.start()
                proc_event.wait(STAGE_GET_TIMEOUT)
                spawned_procs.append(p)
                # stage.write(item)
            # print(out)
            # stage.output_queue.put(out)
        except queue.Empty:
            if (time.time() - keepalive.value) < KEEP_ALIVE_TIMEOUT:
                print('[+] keep alive timeout({}).. killing..'.format(stage.name))
                return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()


class Pipeline:
    def __init__(self):
        self._stages = []
        self._input_queue = MyInputQueue()
        self._keepalive = multiprocessing.Value('d', time.time())
        self._threads_sem = multiprocessing.Semaphore(multiprocessing.cpu_count())
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
        stage.index = len(self._stages)
        self._stages.append(stage)


    def run(self):
        self._keepalive_thread = multiprocessing.Process(target=keepalive_worker,
                                                         args=(self._keepalive, self._stages))
        # self._keepalive_thread.daemon = True
        self._keepalive_thread.start()
        
        ps = []
        for s in self._stages:
            print('[+] intializing process:', s.name)
            target = pipeline_stage_worker
            if s.get_max_parallel() == 1:
                target = no_fork_pipeline_stage_worker
            p = multiprocessing.Process(target=target,
                                        args=(self._threads_sem, s, self._keepalive))
            ps.append(p)
            # p.daemon = True
        print('[+] starting processes')
        [p.start() for p in ps]
        

    def read(self, size):
        self._input_queue.put(size)
        tip = self._get_tip_queue()
        return [tip.get() for a in range(size)]

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
        # # print(data)
        # print('[+] {}::write get_image_with_audio'.format(self.name))
        image = get_image_with_audio(*data)
        # print('[+] {}::write image'.format(self.name), image[0].shape, image[1])
        self.output_queue.put(image)
        # print('[+] {}::write done'.format(self.name))



class ImageToEncoding(PipelineStage):
    def __init__(self):
        super().__init__()
        self._max_parallel = 1

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
        # print('[+] {}::write'.format(self.name))

        img, label = data
        img = np.repeat(img.reshape(224, 224, 1), 3, axis=2)
        # print('[+] {}::write run'.format(self.name))
        embed = self.sess.run(self.vgg.fc1,
                                feed_dict={self.vgg.imgs: [img]})[0]
        # print('[+] {}::write put'.format(self.name))
        self.output_queue.put((embed, label))
        # print('[+] {}::write after put'.format(self.name))


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
        self._stages.append(PrintSummary())

    def read(self, size):
        return super().read(size*2)


class PrintSummary(PipelineStage):
    def __init__(self):
        super().__init__()
        self.finished = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        self.should_display = False

    def write(self, data):
        with self.lock:
            self.finished.value += 1

        print('[+] PrintSummary: {}. {} {}'.format(self.finished.value, data[0].shape, data[1]))

    

class AudioEncoderPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.add_stage(DatasetStage(SimpleDualDS()))
        # self.add_stage(AudioJoinStage())
        # self.add_stage(AudioToImageStage())
        # self.add_stage(ImageToEncoding())
        # self.add_stage(TrainFormatterStage())
        # self.add_stage(PrintSummary())


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
    try:
        import sys
        import multiprocessing
        import platform
        # if 'Darwin' in platform.platform():
            # multiprocessing.set_start_method('spawn')
        if len(sys.argv) < 2:
            read_size = 8
            print('[+] Using default read_size..')
        else:
            read_size = int(sys.argv[1])
            print('[+] Using given read_size:', read_size)

        amp = AudioEncoderPipeline()
        amp.run()
        ret = amp.read(read_size)
    except KeyboardInterrupt:
        if amp._keepalive_thread:
            amp._keepalive_thread.terminate()
        sys.exit(1)
    # import pdb; pdb.set_trace()
