
from xmlrpc.server import SimpleXMLRPCServer
import datetime
import sys

import numpy as np


class MyServer(SimpleXMLRPCServer):
    def __init__(self, *args, **kwargs):
        self.quit = 0
        super().__init__(*args, *kwargs)
        self.register_function(self.kill, 'kill')

    def kill(self):
        print('[+] killing')
        self.quit = 1
        return 1

    def serve_forever(self):
        while not self.quit:
            self.handle_request()
        print('[+] Killed')


class Generator:
    def __init__(self, pipeline=None):
        self.pipeline = None
        self.datasets = None

    def configure(self, config):
        self.datasets = DatasetCollection(config.get('datasets', []))
        

    def get_batch(self, class_name, batch_size):
        raise NotImplementedError()


    def get_zeros(self, num):
        ret = np.zeros(num)
        return ret.tobytes(), ret.dtype.name, ret.shape

# generator = Generator()
# with MyServer(("localhost", 8000)) as server:
#     server.register_instance(pipeline, allow_dotted_names=True)
#     server.register_multicall_functions()
#     print('Serving XML-RPC on localhost port 8000')
#     server.serve_forever()
# generator.configure({
#     'datasets': [
#         {
#             'input': 'audio',
#             'output': 'embed',
#             'path': 'DS_WAV_VOCL',
#             'name': '1',
#         }, {
#             'input': 'audio',
#             'output': 'embed',
#             'path': 'DS_WAV_INST',
#             'name': '2',
#         }
#     ]
# })
from pipeline import AudioMixerPipeline

from pipeline import DatasetStage, AudioMixerStage, PrinterStage
from data import SimpleDualDS


def main():
    import time
    t = time.time()
    amp = AudioMixerPipeline()
    amp.start()
    amp.join()
    print(time.time() - t)

main()
# s = DatasetStage(SimpleDualDS())
# s1 = AudioMixerStage()
# s2 = PrinterStage()
# print(s2.write(s1.write(s.write(64))))
