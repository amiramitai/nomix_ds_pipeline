import redis
import time
import yaml
import sys


class ProfilerEvent:
    def __init__(self, name, conpool):
        self.name = name
        self.start = None
        self.conpool = conpool

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        now = time.time()
        r = redis.StrictRedis(connection_pool=self.conpool)
        r.rpush(self.name, now-self.start)

class Profiler:
    def __init__(self, config):
        self.host = config.get('host')
        self.prefix = config.get('prefix', '')
        self.conpool = redis.ConnectionPool(host=self.host, port=6379, db=0)
        redis.StrictRedis(connection_pool=self.conpool).ping()
        print('[+] profiler initialized successfully!')



    def record(self, name):
        name = '{}{}'.format(self.prefix, name)
        return ProfilerEvent(name, self.conpool)

    def dump(self):
        r = redis.StrictRedis(connection_pool=self.conpool)
        results = {}
        for key in r.scan_iter('{}*'.format(self.prefix)):
            measurements = [float(m) for m in r.lrange(key, 0, -1)]
            cur = {
                'longest': max(measurements),
                'sum': sum(measurements)
            }
            results[key] = cur

        return results



if __name__ == '__main__':
    import yaml
    import sys
    import pprint
    
    if len(sys.argv) < 2:
        print('Usage: python profiler.py <config.yaml>')
        sys.exit(1)
    with open(sys.argv[1]) as f:
        m = yaml.safe_load(f)
        config = m['pipeline']['profiler']
        p = Profiler(config)
        pprint.pprint(p.dump())