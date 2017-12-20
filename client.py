import xmlrpc.client
import numpy as np
import sys


class MyClient:
    def __init__(self):
        server = xmlrpc.client.ServerProxy("http://localhost:8000")
        self.server = server

    def configure(self, config):
        ret = self.server.config(config)
        assert ret == 1

    def get_zeros(self, num):
        buff, dtype, shape = self.server.get_zeros(num)
        return np.frombuffer(buff.data, dtype).reshape(shape)

    def kill(self):
        self.server.kill()

client = MyClient()
# print(client.get_zeros(50))
client.configure({
    'datasets': [
        {
            'input': 'audio',
            'output': 'embed',
            'path': 'DS_WAV_VOCL',
        }, {
            'input': 'audio',
            'output': 'embed',
            'path': 'DS_WAV_INST',
        }
    ]
})
client.kill()
