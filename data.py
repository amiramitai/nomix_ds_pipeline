import multiprocessing
import random
import itertools
import pickle

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
    def __init__(self, filename, _type, _class):
        super().__init__(_type, _class)
        print('[+] caching dataset from file', filename)
        self.f = open(filename, 'r')
        self.filename = filename
        self.line_coords = []
        offset = 0
        for line in self.f:
            self.line_coords.append((offset, len(line)))
            offset += len(line)
        #     if len(self.line_coords) % 1000 == 0:
        #         print(len(self.line_coords), end='\r')
        # print(len(self.line_coords))
        self.f.seek(0)
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
            coord_off = random.randint(0, len(self.line_coords)-1)
            # files.append(self.get_filename_for_coord(coord_off))
            patch = get_audio_patch_with_params(self.get_filename_for_coord(coord_off))
            yield patch.data, self._data_class
        return

    def get_filename_for_coord(self, coord_off):
        with self.lock:
            offset, length = self.line_coords[coord_off]
            self.f.seek(offset)
            ret = self.f.read(length).strip()
        return ret

    def get_length(self):
        return len(self.line_coords)


class SimpleVoclDS(LineDelimFileDataset):
    def __init__(self):
        super().__init__('ds_vocls', DataType.AUDIO, DataClass.VOCAL)


class SimpleInstDS(LineDelimFileDataset):
    def __init__(self):
        super().__init__('ds_inst', DataType.AUDIO, DataClass.INSTRUMENTAL)


class SimpleDualDS:
    def __init__(self):
        self.voclds = SimpleVoclDS()
        self.instds = SimpleInstDS()

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


if __name__ == '__main__':
    # get_audio_patch_with_params('../looperman-a-0064965-0000185-donnievyros-stop-me-cover.mp3')
    # get_audio_patch_with_params('../3rd/vgg16/looperman-a-0933074-0010983-mike0112-run-and-hide-version-1.mp3', 120 + 30)
    get_audio_patch_with_params('../looperman-a-0054911-0001363-jpipes24-vocal-loop-enjoy-the-ride-dry.mp3', 14.19)
    pass

# 0.0034242335 - no
# 0.0334502 - no
# 0.0482797 - no