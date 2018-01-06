import multiprocessing
import random
import itertools
import pickle
import os
from collections import defaultdict


from utils import iterate_files, iterate_audio_files

from audio import AudioFile, get_audio_patch_with_params
from audio import MELS, HOP_LENGTH, SAMPLE_RATE, RMS_SILENCE_THRESHOLD, SILENCE_HOP_THRESHOLD


class DataClass:
    INSTRUMENTAL = 0
    VOCAL = 1


class DataType:
    AUDIO = 0
    IMAGE = 1
    EMBED = 2


class Dataset:
    def __init__(self, _type, _class):
        self._data_type = _type
        self._data_class = _class

    def get_type(self):
        return self._data_type

    def get_class(self):
        return self._data_class

    def get_samples(self, num):
        raise NotImplementedError()

class LineDelimFileDataset(Dataset):
    def __init__(self, filename, base_dir, _type, _class):
        super().__init__(_type, _class)
        print('[+] caching dataset from file', filename)
        self.filename = filename
        self.line_coords = []
        self.base_dir = base_dir
        offset = 0
        with open(filename, 'rb') as f:
            for line in f:
                self.line_coords.append((offset, len(line)))
                offset += len(line)
        #     if len(self.line_coords) % 1000 == 0:
        #         print(len(self.line_coords), end='\r')
        # print(len(self.line_coords))
        # self.f.seek(0)
        self.coord_ptr = 0
        self.lock = multiprocessing.Lock()

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
            patch = get_audio_patch_with_params(filename)
            yield patch.data, self._data_class
        return

    def get_rand_filename(self):
        coord_off = random.randint(0, len(self.line_coords)-1)
        return self.get_filename_for_coord(coord_off)

    def get_filename_for_coord(self, coord_off):
        with self.lock, open(self.filename, 'rb') as f:
            offset, length = self.line_coords[coord_off]
            f.seek(offset)
            ret = f.read(length).strip().decode('utf-8')
            if self.base_dir:
                return os.path.join(self.base_dir, ret)
            return ret

    def get_length(self):
        return len(self.line_coords)


class SimpleDualDS:
    def __init__(self, params):
        # if not params:
        #     params = {}
        vocl_fn = params.get('vocl_filename', 'ds_vocls')
        inst_fn = params.get('inst_filename', 'ds_inst')
        base_dir = params.get('base_dir')
        self.voclds = LineDelimFileDataset(vocl_fn, base_dir, DataType.AUDIO, DataClass.VOCAL)
        self.instds = LineDelimFileDataset(inst_fn, base_dir, DataType.AUDIO, DataClass.INSTRUMENTAL)

    def read(self, num):
        if num % 2 != 0:
            raise RuntimeError('num must be even', num)
        v = self.voclds.read(int(num/2))
        i = self.instds.read(int(num/2))
        return itertools.chain(v, i)


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

        for a in iterate_files(os.path.join(path, 'Mixtures'), '.wav'):
            key = os.path.basename(os.path.dirname(a))
            self.samples[key]['mix'] = a
        
        for a in iterate_files(os.path.join(path, 'Sources'), '.wav'):
            key = os.path.basename(os.path.dirname(a))
            if a.endswith('vocals.wav'):
                self.samples[key]['vocl'] = a
            else:
                self.samples[key]['inst'].append(a)


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
            # elif a.endswith('source-02.wav'):
                self.samples[key]['vocl'] = a
            # else:
            #     raise RuntimeError('Unknown file', a)





if __name__ == '__main__':
    from pprint import pprint
    # get_audio_patch_with_params('../looperman-a-0064965-0000185-donnievyros-stop-me-cover.mp3')
    # get_audio_patch_with_params('../3rd/vgg16/looperman-a-0933074-0010983-mike0112-run-and-hide-version-1.mp3', 120 + 30)
    # get_audio_patch_with_params('../looperman-a-0054911-0001363-jpipes24-vocal-loop-enjoy-the-ride-dry.mp3', 14.19)
    # ld = LineDelimFileDataset(r'T:\datasets\nomix_ds\ds_vocls', r'T:\datasets\nomix_ds', DataType.AUDIO, DataClass.VOCAL)
    # pprint(DSD100({'path':'/Volumes/t$/datasets/DSD100'}).samples)
    pprint(CCMixter({'path':'/Volumes/t$/datasets/ccmixter'}).samples)

    # import pdb; pdb.set_trace()

# 0.0034242335 - no
# 0.0334502 - no
# 0.0482797 - no