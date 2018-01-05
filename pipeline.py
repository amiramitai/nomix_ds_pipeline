import itertools
import multiprocessing
import time
import os
import ctypes
import yaml

import pipeline_stage
from myqueue import MyInputQueue

from context import Context
from profiler import Profiler
from pipeline_workers import pipeline_stage_worker,\
                             no_fork_pipeline_stage_worker,\
                             keepalive_worker

from pydub import AudioSegment
from utils import AsyncGenerator, StringValue

class Pipeline:
    def __init__(self):
        self._stages = []
        self._input_queue = MyInputQueue()
        self._threads_sem = multiprocessing.Semaphore(multiprocessing.cpu_count())
        self._keepalive_thread = None
        self._cache_folder = None
        self._context = Context()

    def get_samples(self, dtype, num):
        raise NotImplementedError()

    def _get_tip_queue(self):
        if not self._stages:
            return self._input_queue
        
        return self._stages[-1].output_queue

    def add_stage(self, stage):
        tip = self._get_tip_queue()
        stage.input_queue = tip
        stage.index = len(self._stages)
        self._stages.append(stage)


    def run(self):
        # make sure all threads are initialized before start working
        init_barrier = multiprocessing.Barrier(len(self._stages) + 1)
        ps = []
        for s in self._stages:
            print('[+] intializing process:', s.name)
            target = pipeline_stage_worker
            if s.get_max_parallel() == 1:
                target = no_fork_pipeline_stage_worker
            kwargs = {
                'init_barrier': init_barrier,
                'thread_sem': self._threads_sem,
                'stage': s,
                'context': self._context,
            }
            p = multiprocessing.Process(target=target, kwargs=kwargs)
            ps.append(p)
        print('[+] starting processes')
        [p.start() for p in ps]
        init_barrier.wait()

    @property
    def name(self):
        return self.__class__.__name__

    def iterate(self, size):
        while True:
            ret = self.read(size)
            yield ret

    def keepalive(self, async=False, timeout=None):
        self._input_queue.put(1)
        self._keepalive_thread = multiprocessing.Process(target=keepalive_worker,
                                                         args=(self._context,
                                                               self._stages))
        self._keepalive_thread.daemon = True
        self._keepalive_thread.start()
        if async:
            return
        
        self._keepalive_thread.join(timeout)


class AudioMixerPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self._stages.append(DualDatasetStage())
        self._stages.append(AudioMixerStage())
        self._stages.append(AudioToImageStage())
        self._stages.append(ImageToEncoding())
        self._stages.append(PrintSummary())

    def read(self, size):
        return super().read(size*2)


class AudioEncoderPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.add_stage(DualDatasetStage())
        # self.add_stage(AudioJoinStage())
        self.add_stage(AudioToImageStage())
        self.add_stage(ImageToEncodingStage())
        # self.add_stage(TrainFormatterStage())
        # self.add_stage(PrintSummary())

class NomixConfigFactory:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            self.map = yaml.safe_load(f)
        self._stages = {}
        self._pl = Pipeline()
        self._load_stages_with_map(self.map['pipeline']['stages'])
        self._load_profiler(self.map['pipeline'].get('profiler'))

    def _load_profiler(self, profiler):
        if not profiler:
            return
        
        self._pl._context.profiler = Profiler(profiler)

    def _load_stages_with_map(self, stages):
        for stage in stages:
            _cls = getattr(pipeline_stage, stage['type'])
            inst = _cls(stage)
            self._pl.add_stage(inst)

    def get_pipeline(self):
        return self._pl
        

if __name__ == '__main__':
    try:
        import sys
        import multiprocessing
        import platform
        if len(sys.argv) < 2:
            print('Usage: python pipeline.py <config.yaml>')
            sys.exit(1)

        amp = NomixConfigFactory(sys.argv[1]).get_pipeline()
        # amp = AudioEncoderPipeline()
        # amp.set_cache('/Users/amiramitai/cache')
        amp.run()
        # ret = amp.read(16)
        amp.keepalive()
    except KeyboardInterrupt:
        if amp._keepalive_thread:
            amp._keepalive_thread.terminate()
        sys.exit(1)
    # import pdb; pdb.set_trace()
