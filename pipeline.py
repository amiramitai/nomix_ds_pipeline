import itertools
import multiprocessing
import threading
import traceback
import time
import os
import ctypes
import pickle
from xmlrpc.server import SimpleXMLRPCServer

import yaml

import pipeline_stage
from myqueue import MyInputQueue, MyQueue

from context import Context
from profiler import Profiler
from pipeline_workers import pipeline_stage_worker,\
                             no_fork_pipeline_stage_worker,\
                             keepalive_worker,\
                             remote_pipe_send_worker

from pydub import AudioSegment
from utils import AsyncGenerator, StringValue

class Pipeline:
    def __init__(self, config):
        self._stages = []
        self._input_queue = MyInputQueue()
        self._keepalive_thread = None
        self._remote_pipe_thread = None
        self._cache_folder = None
        self._context = Context()
        self._config = config
        self._server = None
        self._server_mode = None
        self._load_server(config.get('server'))
        self._load_remotes(config.get('remotes', []))
        self._load_profiler(config.get('profiler'))
        self._load_stages_with_config(config['stages'])

    def _load_server(self, server_config):
        if not server_config:
            return
        host = server_config['host']
        port = server_config['port']
        self._server_mode = server_config.get('mode')
        print('[+] listening on {}:{}'.format(host, port))
        self._server = SimpleXMLRPCServer((host, port), logRequests=False)
        self._server.register_function(self._put_in_pipe, 'put_remote')

    def _put_in_pipe(self, name, item):
        try:
            # print('[+] _put_in_pipe', name)
            for s in self._stages:
                # print('[+] _put_in_pipe loop:', s.name, name)
                if s.name == name:
                    # print('[+] _put_in_pipe::putting in {}'.format(name))
                    s.output_queue.put(pickle.loads(item.data))
                    # print('[+] _done in {}'.format(name))
                    return 0
            # print('[+] _put_in_pipe::found no pipe to put in.. dropping..')
            return 0
        except:
            traceback.print_exc()
            raise

    def _load_profiler(self, profiler):
        if not profiler:
            return
        
        self._context.profiler = Profiler(profiler)

    def _load_remotes(self, remotes_config):
        for remote in remotes_config:
            name = remote['name']
            self._context.remotes[name] = MyQueue()

    def _load_stages_with_config(self, stages_config):
        for sc in stages_config:
            _cls = getattr(pipeline_stage, sc['type'])
            self.add_stage(_cls(sc, self._context))

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

    def _serve_forever(self):
        print('[+] serving...')
        while True:
            self._server.handle_request()


    def run(self):
        if 'remotes' in self._config:
            self._remote_pipe(block=False)

        if self._server:
            threading.Thread(target=self._serve_forever).start()
            
        if self._server_mode == 'idle':
            print('[!] going idle mode')
            return

        # make sure all threads are initialized before start working
        init_barrier = multiprocessing.Barrier(len(self._stages) + 1)
        thread_sem = multiprocessing.BoundedSemaphore(multiprocessing.cpu_count())
        ps = []
        for s in self._stages:
            print('[+] intializing process:', s.name)
            target = pipeline_stage_worker
            if s.get_max_parallel() == 1:
                target = no_fork_pipeline_stage_worker
            kwargs = {
                'init_barrier': init_barrier,
                'thread_sem': thread_sem,
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

    def _remote_pipe(self, block=False, timeout=None):
        self._remote_pipe_thread = multiprocessing.Process(target=remote_pipe_send_worker,
                                                           args=(self._config, self._context))
        self._remote_pipe_thread.daemon = True
        self._remote_pipe_thread.start()
        if not block:
            return
        
        self._remote_pipe_thread.join(timeout)
    
    
    def keepalive(self, block=False, timeout=None):
        self._input_queue.put(1)
        self._keepalive_thread = multiprocessing.Process(target=keepalive_worker,
                                                         args=(self._context,
                                                               self._stages))
        self._keepalive_thread.daemon = True
        self._keepalive_thread.start()
        if not block:
            return
        
        self._keepalive_thread.join(timeout)


# class AudioMixerPipeline(Pipeline):
#     def __init__(self):
#         super().__init__()
#         self._stages.append(DualDatasetStage())
#         self._stages.append(AudioMixerStage())
#         self._stages.append(AudioToImageStage())
#         self._stages.append(ImageToEncoding())
#         self._stages.append(PrintSummary())

#     def read(self, size):
#         return super().read(size*2)


# class AudioEncoderPipeline(Pipeline):
#     def __init__(self):
#         super().__init__()
#         self.add_stage(DualDatasetStage())
#         # self.add_stage(AudioJoinStage())
#         self.add_stage(AudioToImageStage())
#         self.add_stage(ImageToEncodingStage())
#         # self.add_stage(TrainFormatterStage())
#         # self.add_stage(PrintSummary())
        

if __name__ == '__main__':
    try:
        import sys
        import multiprocessing
        import platform
        if len(sys.argv) < 2:
            print('Usage: python pipeline.py <config.yaml>')
            sys.exit(1)

        yaml_path = sys.argv[1]
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        amp = Pipeline(config['pipeline'])
        import pdb
        pdb.set_trace()
        sdfsdf
        amp.run()
        amp.keepalive(block=True)
    except KeyboardInterrupt:
        if amp._keepalive_thread:
            amp._keepalive_thread.terminate()
        sys.exit(1)
    # import pdb; pdb.set_trace()
