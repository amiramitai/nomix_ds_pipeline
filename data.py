import multiprocessing
import random
import itertools
import pickle
import os
import re
from collections import defaultdict


from utils import iterate_files, iterate_audio_files

from audio import get_net_duration
from audio import MELS, HOP_LENGTH, SAMPLE_RATE, RMS_SILENCE_THRESHOLD, SILENCE_HOP_THRESHOLD


class DataClass:
    INSTRUMENTAL = 0
    VOCAL = 1


class DataType:
    AUDIO = 0
    IMAGE = 1
    EMBED = 2


class LineDelimFileDataset:
    def __init__(self, filename, base_dir):
        print('[+] caching dataset from file', filename)
        self.filename = filename
        self.line_coords = []
        self.base_dir = base_dir
        offset = 0
        with open(filename, 'rb') as f:
            for line in f:
                self.line_coords.append((offset, len(line)))
                offset += len(line)

        self.coord_ptr = 0

    def shuffle(self):
        print('[+] shuffling:', self.filename)
        random.shuffle(self.line_coords)
        self.coord_ptr = 0

    def is_eof(self):
        return self.coord_ptr >= (len(self.line_coords) - 1)
    
    def read(self, batch_size):
        files = []
        for i in range(batch_size):
            filename = self.get_rand_filename()
            # files.append(self.get_filename_for_coord(coord_off))
            # patch = get_audio_patch_with_params(filename)
            yield patch.data
        return

    def samples(self):
        line_coords = self.line_coords[:]
        random.shuffle(line_coords)
        for coord in line_coords:
            filename = self.get_filename_for_coord(coord)
            yield os.path.join(self.base_dir, filename)


    def get_rand_filename(self):
        coord = random.choice(self.line_coords)
        return self.get_filename_for_coord(coord)

    def get_filename_for_coord(self, coord):
        with open(self.filename, 'rb') as f:
            offset, length = coord
            f.seek(offset)
            ret = f.read(length).strip().decode('utf-8')
            if self.base_dir:
                return os.path.join(self.base_dir, ret)
            return ret

    def get_length(self):
        return len(self.line_coords)


class NomixDS:
    def __init__(self, params):
        # if not params:
        #     params = {}
        vocl_fn = params.get('vocl_filename', 'ds_vocls')
        inst_fn = params.get('inst_filename', 'ds_inst')
        base_dir = params.get('base_dir')
        self.voclds = LineDelimFileDataset(vocl_fn, base_dir)
        self.instds = LineDelimFileDataset(inst_fn, base_dir)

    def vocals(self):
        while True:
            samples = self.voclds.samples()
            for sample in samples:
                yield sample

            

    def instrumentals(self):
        while True:
            samples = self.instds.samples()
            for sample in samples:
                yield sample
    


class DatasetCollection:
    def __init__(self, datasets):
        self._datasets = []
        for ds in datasets:
            if isinstance(ds, dict):
                ds = Dataset(ds)
            elif not isinstance(ds, Dataset):
                raise RuntimeError('Expected a Dataset', ds)
            self._datasets.append(ds)

    def filter_dataset_type(self, dtype):
        ret = [d for d in self._datasets if d.get_type() == dtype]
        return DatasetCollection(ret)

    def filter_dataset_class(self, _class):
        ret = [d for d in self._datasets if d.get_class() == _class]
        return DatasetCollection(ret)

    def get_types(self, dtype):
        return list(set([ds.get_type() for ds in self._datasets]))

    def get_classes(self, dtype):
        return list(set([ds.get_classes() for ds in self._datasets]))


class DataSource:
    def __init__(self):
        raise NotImplementedError()


class DSD100:
    def __init__(self, params):
        self.samples = defaultdict(lambda: {'mix': None, 'vocl': None, 'inst': []})
        path = params['path']

        self.number_if_samples = 0
        for a in iterate_files(os.path.join(path, 'Mixtures'), '.wav'):
            print(a)
            key = os.path.basename(os.path.dirname(a))
            self.samples[key]['mix'] = a
            self.number_if_samples += 1

        
        for a in iterate_files(os.path.join(path, 'Sources'), '.wav'):
            key = os.path.basename(os.path.dirname(a))
            if a.endswith('vocals.wav'):
                self.samples[key]['vocl'] = a
            else:
                self.samples[key]['inst'].append(a)
            self.number_if_samples += 1

        self.samples = dict(self.samples)


    def mixtures(self):
        items = list(self.samples.values())
        random.shuffle(items)
        for item in items:
            if item['mix'] and item['vocl']:
                yield item['mix'], item['vocl']
    
    def vocals(self):
        items = list(self.samples.values())
        random.shuffle(items)
        for item in items:
            if item['vocl']:
                yield item['vocl']

    def instrumentals(self):
        items = list(self.samples.values())
        random.shuffle(items)
        for item in items:
            for inst in item['inst']:
                yield inst
    



class CCMixter:
    def __init__(self, params):
        self.samples = defaultdict(lambda: {'mix': None, 'vocl': None, 'inst': None})
        path = params['path']

        for a in iterate_files(path, '.wav'):
            key = os.path.basename(os.path.dirname(a))
            if a.endswith('mix.wav'):
                self.samples[key]['mix'] = a
            elif a.endswith('source-01.wav'):
                self.samples[key]['inst'] = a
            else:
                self.samples[key]['vocl'] = a
        self.samples = dict(self.samples)

    def mixtures(self):
        items = list(self.samples.values())
        random.shuffle(items)
        for item in items:
            if item['mix']:
                yield item['mix']
    
    def vocals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            if item['vocl']:
                yield item['vocl']

    def instrumentals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            if item['inst']:
                yield item['inst']

class Irmas:
    def __init__(self, params):
        path = params['path']
        self.vocl = []
        self.inst = []
        reg = re.compile('\[(.*?)\]')
        for a in iterate_files(path, '.wav'):
            try:
                txt_filename = a[:-4] + '.txt'
                if 'voi' in map(str.strip, open(txt_filename)):
                    self.vocl.append(a)
                else:
                    self.inst.append(a)
            except FileNotFoundError:
                if 'voi' in reg.findall(a):
                    self.vocl.append(a)
                else:
                    self.inst.append(a)

    def vocals(self):
        items = self.vocl[:]
        random.shuffle(items)
        for item in items:
            yield item

    def instrumentals(self):
        items = self.inst[:]
        random.shuffle(items)
        for item in items:
            yield item
        

class JamAudio:
    def __init__(self, params):
        self.path = params['path']
        self.samples = defaultdict(lambda: {'sing': [], 'nosing': []})

        for a in iterate_audio_files(self.path):
            print(a)
            key = os.path.basename(a)
            for line in map(str.strip, open(a[:-4] + '.lab')):
                frm, to, label = line.split()
                self.samples[key][label].append((float(frm), float(to)))

            self.samples[key]

        self.samples = dict(self.samples)

    def mixtures(self):
        return
        yield

    def vocals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            filepath = os.path.join(self.path, key)
            if item['sing']:
                yield filepath, random.choice(item['sing'])

    def instrumentals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            filepath = os.path.join(self.path, key)
            if item['nosing']:
                yield filepath, random.choice(item['nosing'])


class Musdb18:
    def __init__(self, params):
        self.samples = defaultdict(lambda: {'mix': None, 'vocl': None, 'inst': []})
        path = params['path']

        for a in iterate_files(path, '.wav'):
            key = os.path.basename(a)[:-6]
            if a.endswith('_4.wav'):
                self.samples[key]['vocl'] = a
            elif a.endswith('_0.wav'):
                self.samples[key]['mix'] = a
            else:
                self.samples[key]['inst'].append(a)

        self.samples = dict(self.samples)

    def mixtures(self):
        items = list(self.samples.values())
        random.shuffle(items)
        for item in items:
            if item['mix'] and item['vocl']:
                yield item['mix'], item['vocl']
    
    def vocals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            if item['vocl']:
                yield item['vocl']

    def instrumentals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            for inst in item['inst']:
                yield inst


class Quasi:
    VOCAL_KEYWORDS = ['choir', 'speech', 'lv_', 'harmo', 'vox', 'voix', 'voic', 'voc']
    
    def __init__(self, params):
        self.samples = defaultdict(lambda: {'mix': [], 'vocl': [], 'inst': []})
        path = params['path']
        self.net_vocals = 0
        self.net_insts = 0
        for a in iterate_files(os.path.join(path, 'separation'), '.wav'):
            print(a)
            key = os.path.basename(os.path.dirname(os.path.dirname(a))).lower()
            filename = os.path.basename(a).lower()
            if self._is_vocal_name(filename):
                self.samples[key]['vocl'].append(a)
                self.net_insts += get_net_duration(a)
                continue
            if 'mix' in filename:
                self.samples[key]['mix'].append(a)
                continue

            self.samples[key]['inst'].append(a)
            self.net_insts += get_net_duration(a)
        self.samples = dict(self.samples)

    def vocals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            sample = self.samples[key]
            # for mix in sample['mix']:
            #     yield mix
            for vocl in sample['vocl']:
                yield vocl

    def instrumentals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            sample = self.samples[key]
            for inst in sample['inst']:
                yield inst
        



    @classmethod
    def _is_vocal_name(cls, name):
        for kw in cls.VOCAL_KEYWORDS:
            if kw in name:
                return True
        return False


class MultiDatasets:
    def __init__(self, params):
        print('[+] MultiDatasets::__init__:', params)
        self.datasets = []
        self.params = []
        for ds_config in params:
            self._load_dataset_with_config(ds_config)
        print('[+] {} datasets were loaded'.format(len(self.datasets)))

    def _load_dataset_with_config(self, config):
        print('[+] MultiDatasets::_load_dataset_with_config:', config)
        cache = config.get('cache')
        if cache and os.path.isfile(cache):
            print('[+] getting from cache!', cache)
            self.datasets.append(pickle.load(open(cache, 'rb')))
            return
        
        _cls = globals()[config['type']]
        inst = _cls(config['params'])
        self.datasets.append(inst)

        if cache:
            pickle.dump(inst, open(cache, 'wb'))

    def vocals(self):
        return self._iterate_iterators(self._get_all_vocals_iterators, label=[0, 1])
    
    def instrumentals(self):
        return self._iterate_iterators(self._get_all_instrumental_iterators, label=[1, 0])

    def _get_all_vocals_iterators(self):
        return [d.vocals() for d in self.datasets]
    
    def _get_all_instrumental_iterators(self):
        return [d.instrumentals() for d in self.datasets]

    def _iterate_iterators(self, iter_func, label):
        while True:
            iters = iter_func()
            while iters:
                random.shuffle(iters)
                ended = []
                
                for i, iterator in enumerate(iters):
                    try:
                        x = next(iterator)
                        # print('[+] iterating', x, label)
                        yield x, label
                    except StopIteration:
                        ended.append(i)
                
                for i in reversed(sorted(ended)):
                    del iters[i]
                    # print('[+] iterator has finished for', i, label)
            
            # print('[+] finished all iterators')
                

if __name__ == '__main__':
    from pprint import pprint
    # get_audio_patch_with_params('../looperman-a-0064965-0000185-donnievyros-stop-me-cover.mp3')
    # get_audio_patch_with_params('../3rd/vgg16/looperman-a-0933074-0010983-mike0112-run-and-hide-version-1.mp3', 120 + 30)
    # get_audio_patch_with_params('../looperman-a-0054911-0001363-jpipes24-vocal-loop-enjoy-the-ride-dry.mp3', 14.19)
    # ld = LineDelimFileDataset(r'T:\datasets\nomix_ds\ds_vocls', r'T:\datasets\nomix_ds', DataType.AUDIO, DataClass.VOCAL)
    # d = DSD100({'path':'/Volumes/t$/datasets/DSD100'})
    import pickle
    d = pickle.loads(open('dsd', 'rb').read())
    d.get_sample(DataClass.VOCAL, {'length': 2.7})
    import pdb; pdb.set_trace()
    # pprint(CCMixter({'path':'/Volumes/t$/datasets/ccmixter'}).samples)
    # irmas = Irmas({'path':'/Volumes/t$/datasets/irmas'})
    # pprint(irmas.vocl)
    # pprint(irmas.inst)
    # pprint(JamAudio({'path':'/Volumes/t$/datasets/jam_audio'}).samples)
    # pprint(Musdb18({'path':'/Volumes/t$/datasets/musdb18'}).samples)
    # pprint(Quasi({'path':'/Volumes/d$/nomix_data/datasets/QUASI'}).samples)
    


    # import pdb; pdb.set_trace()

# 0.0034242335 - no
# 0.0334502 - no
# 0.0482797 - no