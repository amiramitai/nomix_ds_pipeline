import multiprocessing
import random
import itertools
import pickle
import os
import re
import traceback
from collections import defaultdict

import numpy as np


from utils import iterate_files, iterate_audio_files

from audio import get_net_duration, get_number_of_frames, get_offset_range_patch
from audio import get_spect_range_from_time_range, choose_spect_range
from audio import HOP_LENGTH, SAMPLE_RATE, RMS_SILENCE_THRESHOLD, SILENCE_HOP_THRESHOLD
from audio import SAMPLE_MIN_LENGTH

from collections import defaultdict


FCN_VOC_THRESHOLD = 0.5

class DataClass:
    INSTRUMENTAL = 0
    VOCAL = 1


class DataType:
    AUDIO = 0
    IMAGE = 1
    EMBED = 2


class DatasetResult:
    pass

class InstWithVocalResult(DatasetResult):
    def __init__(self, vocal_res, inst_res):
        self.voc = vocal_res
        self.inst = inst_res

    def mix(self):
        params, _ = self.inst
        filename, offset, _range = params
        iaud = get_offset_range_patch(filename, offset, _range)
        params, _ = self.voc
        filename, offset, _range = params
        vaud = get_offset_range_patch(filename, offset, _range)
        return (vaud + iaud) / 2.0

    def desc(self):
        return 'voc:' + str(self.voc[0]) + '+ inst:' + str(self.inst[0])

    def get_label(self):
        return self.voc[1]

    def get_fcn_label(self):
        params, _ = self.inst
        filename, offset, _range = params
        others = get_offset_range_patch(filename, offset, _range)
        others = (others > FCN_VOC_THRESHOLD).astype('float')
        others = others.reshape((others.shape[0], others.shape[1], 1))

        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1))

        return np.concatenate((others, voc), axis=2)

    def get_frrn_label(self):
        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        # voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1))

        # return voc / 2.0
        return (voc > FCN_VOC_THRESHOLD).astype('int32')
    
    def get_frrn2_label(self):
        params, _ = self.inst
        filename, offset, _range = params
        others = get_offset_range_patch(filename, offset, _range)
        # others = (others > FCN_VOC_THRESHOLD).astype('float')
        others = others.reshape((others.shape[0], others.shape[1], 1)) / 2.0

        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        # voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1)) / 2.0

        return np.concatenate((others, voc), axis=2)
    
    def get_rnn_label(self):
        params, _ = self.inst
        filename, offset, _range = params
        others = get_offset_range_patch(filename, offset, _range)
        # others = (others > FCN_VOC_THRESHOLD).astype('float')
        others = others.reshape((others.shape[0], others.shape[1], 1)) / 2.0

        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        # voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1)) / 2.0

        return others, voc


class MixWithVocalResult(DatasetResult):
    def __init__(self, vocal_res, mix_filename):
        self.voc = vocal_res
        self.mix = mix_filename

    def _slice(self):
        params, _ = self.voc
        filename, offset, _range = params
        ret = get_offset_range_patch(filename, offset, _range, self.mix)
        return ret

    def desc(self):
        return 'voc:' + str(self.voc[0][0]) + '+ mix:' + self.mix

    def get_label(self):
        return self.voc[1]

    def get_fcn_label(self):
        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        mix = get_offset_range_patch(filename, offset, _range, self.mix)
        others = np.clip(mix - voc, 0, 1)

        voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        others = (others > FCN_VOC_THRESHOLD).astype('float')
        
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1))
        others = others.reshape((others.shape[0], others.shape[1], 1))
        return np.concatenate((others, voc), axis=2)

    def get_frrn_label(self):
        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        # voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1))
        return (voc > FCN_VOC_THRESHOLD).astype('int32')
    
    def get_frrn2_label(self):
        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        mix = get_offset_range_patch(filename, offset, _range, self.mix)
        others = np.clip(mix - voc, 0, 1)

        # voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        # others = (others > FCN_VOC_THRESHOLD).astype('float')
        
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1))
        others = others.reshape((others.shape[0], others.shape[1], 1))
        return np.concatenate((others, voc), axis=2)

    def get_rnn_label(self):
        params, _ = self.voc
        filename, offset, _range = params
        voc = get_offset_range_patch(filename, offset, _range)
        mix = get_offset_range_patch(filename, offset, _range, self.mix)
        others = np.clip(mix - voc, 0, 1)

        # voc = (voc > FCN_VOC_THRESHOLD).astype('float')
        # others = (others > FCN_VOC_THRESHOLD).astype('float')
        
        voc = voc.reshape((voc.shape[0], voc.shape[1], 1))
        others = others.reshape((others.shape[0], others.shape[1], 1))
        return others, voc

        

class JustVocalResult(DatasetResult):
    def __init__(self, filename, offset, _range=None):
        self.filename = filename
        self.offset = offset
        self._range = _range


class JustInstResult(DatasetResult):
    def __init__(self, filename, offset, _range=None):
        self.filename = filename
        self.offset = offset
        self._range = _range


class Dataset:
    def __init__(self):
        self.ranges = defaultdict(int)
        self.file_ranges = defaultdict(list)

    def add_frames_num(self, name, filename):
        frames = get_number_of_frames(filename)
        self.ranges[name] += frames
        self.file_ranges[name].append((frames, filename))

    def get_frames_num(self, name):
        return self.ranges.get(name, 0)

    def get_with_perm(self, label_name, offset):
        for frame_num, filename in self.file_ranges[label_name]:
            if offset <= frame_num:
                return filename, offset, None
            
            offset -= frame_num

    def get_mixture_with_vocal(self, filename):
        return None


class LineDelimFileDataset:
    def __init__(self, filename, base_dir, file_cb):
        super().__init__()
        print('[+] caching dataset from file', filename)
        self.filename = filename
        self.line_coords = []
        self.base_dir = base_dir
        offset = 0
        with open(filename, 'r') as f:
            for line in f:
                fullpath = os.path.join(self.base_dir, line.strip())
                file_cb(fullpath)
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


class NomixDS(Dataset):
    def __init__(self, params):
        super().__init__()
        # if not params:
        #     params = {}
        vocl_fn = params.get('vocl_filename', 'ds_vocls')
        inst_fn = params.get('inst_filename', 'ds_inst')
        base_dir = params.get('base_dir')
        
        self.voclds = LineDelimFileDataset(vocl_fn, base_dir, self._add_vocals)
        # self.ranges['vcl'] = self.voclds.get_frames_num('any')
        self.instds = LineDelimFileDataset(inst_fn, base_dir, self._add_instrumentals)
        # self.ranges['inst'] = self.instds.get_frames_num('any')

    def _add_vocals(self, filename):
        self.add_frames_num('vocl', filename)
    
    def _add_instrumentals(self, filename):
        self.add_frames_num('inst', filename)

    def vocals(self):
        samples = self.voclds.samples()
        for sample in samples:
            yield sample

            

    def instrumentals(self):
        samples = self.instds.samples()
        for sample in samples:
            yield sample


class DSD100(Dataset):
    def __init__(self, params):
        super().__init__()
        self.samples = defaultdict(lambda: {'mix': None, 'vocl': None, 'inst': []})
        path = params['path']

        self.number_if_samples = 0
        for a in iterate_files(os.path.join(path, 'Mixtures'), '.wav'):
            key = os.path.basename(os.path.dirname(a))
            self.samples[key]['mix'] = a
            self.number_if_samples += 1

        
        for a in iterate_files(os.path.join(path, 'Sources'), '.wav'):
            key = os.path.basename(os.path.dirname(a))
            if a.endswith('vocals.wav'):
                self.samples[key]['vocl'] = a
                self.add_frames_num('vocl', a)
            else:
                self.samples[key]['inst'].append(a)
                self.add_frames_num('inst', a)
            self.number_if_samples += 1

        self.samples = dict(self.samples)


    # def mixtures(self):
    #     items = list(self.samples.values())
    #     random.shuffle(items)
    #     for item in items:
    #         if item['mix'] and item['vocl']:
    #             yield item['mix'], item['vocl']

    def get_mixture_with_vocal(self, filename):
        key = os.path.basename(os.path.dirname(filename))
        if key not in self.samples:
            print('[+] DSD100::get_mixture_with_vocal:: key was not found in samples', key)
            return None
        val = self.samples[key]
        if 'mix' not in val:
            print('[+] DSD100::get_mixture_with_vocal:: mix was not found in val', val.keys())
            return None

        return val['mix']
    
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


class CCMixter(Dataset):
    def __init__(self, params):
        super().__init__()
        self.samples = defaultdict(lambda: {'mix': None, 'vocl': None, 'inst': None})
        path = params['path']

        for a in iterate_files(path, '.wav'):
            key = os.path.basename(os.path.dirname(a))
            if a.endswith('mix.wav'):
                self.samples[key]['mix'] = a
            elif a.endswith('source-01.wav'):
                self.samples[key]['inst'] = a
                self.add_frames_num('inst', a)
            else:
                self.samples[key]['vocl'] = a
                self.add_frames_num('vocl', a)
        self.samples = dict(self.samples)

    # def mixtures(self):
    #     items = list(self.samples.values())
    #     random.shuffle(items)
    #     for item in items:
    #         if item['mix']:
    #             yield item['mix']

    def get_mixture_with_vocal(self, filename):
        key = os.path.basename(os.path.dirname(filename))
        if key not in self.samples:
            print('[+] CCMixter::get_mixture_with_vocal:: key was not found in samples', key)
            return None
        val = self.samples[key]
        if 'mix' not in val:
            print('[+] CCMixter::get_mixture_with_vocal:: mix was not found in val', val.keys())
            return None

        return val['mix']
    
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


class Irmas(Dataset):
    def __init__(self, params):
        super().__init__()
        path = params['path']
        self.vocl = []
        self.inst = []
        reg = re.compile('\[(.*?)\]')
        for a in iterate_files(path, '.wav'):
            try:
                txt_filename = a[:-4] + '.txt'
                if 'voi' in map(str.strip, open(txt_filename)):
                    self.vocl.append(a)
                    self.add_frames_num('vocl', a)
                else:
                    self.inst.append(a)
                    self.add_frames_num('inst', a)
            except FileNotFoundError:
                if 'voi' in reg.findall(a):
                    self.vocl.append(a)
                    self.add_frames_num('vocl', a)
                else:
                    self.inst.append(a)
                    self.add_frames_num('inst', a)
    
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
        

class JamAudio(Dataset):
    def __init__(self, params):
        super().__init__()
        self.path = params['path']
        self.samples = defaultdict(lambda: {'sing': [], 'nosing': []})

        for a in iterate_audio_files(self.path):
            key = os.path.basename(a)
            vocl_frames = 0
            inst_frames = 0
            for line in map(str.strip, open(a[:-4] + '.lab')):
                frm, to, label = line.split()
                spect_range = get_spect_range_from_time_range((float(frm), float(to)))
                self.samples[key][label].append(spect_range)
                start, end = spect_range
                
                if end - start < SAMPLE_MIN_LENGTH:
                    continue
                
                frames = (end - start) - SAMPLE_MIN_LENGTH + 1
                if label == 'sing':
                    vocl_frames += frames
                elif label == 'nosing':
                    inst_frames += frames
                else:
                    print('[!] unknown key:', key)

            if vocl_frames > 0:
                self.ranges['vocl'] += vocl_frames
                self.file_ranges['vocl'].append((vocl_frames, a))
            
            if inst_frames > 0:
                self.ranges['inst'] += inst_frames
                self.file_ranges['inst'].append((inst_frames, a))

        self.samples = dict(self.samples)

    def get_with_perm(self, label_name, offset):
        klabel = 'nosing'
        if label_name == 'vocl':
            klabel = 'sing'

        for frame_num, filename in self.file_ranges[label_name]:
            if offset <= frame_num:
                key = os.path.basename(filename)
                return filename, offset, self.samples[key][klabel]

            offset -= frame_num

    def vocals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            filepath = os.path.join(self.path, key)
            if item['sing']:
                yield filepath, choose_spect_range(item['sing'])

    def instrumentals(self):
        keys = list(self.samples.keys())
        random.shuffle(keys)
        for key in keys:
            item = self.samples[key]
            filepath = os.path.join(self.path, key)
            if item['nosing']:
                yield filepath, choose_spect_range(item['nosing'])


class Musdb18(Dataset):
    def __init__(self, params):
        super().__init__()
        self.samples = defaultdict(lambda: {'mix': None, 'vocl': None, 'inst': []})
        path = params['path']

        for a in iterate_files(path, '.wav'):
            key = os.path.basename(a)[:-6]
            if a.endswith('_4.wav'):
                self.samples[key]['vocl'] = a
                self.add_frames_num('vocl', a)
            elif a.endswith('_0.wav'):
                self.samples[key]['mix'] = a
            else:
                self.samples[key]['inst'].append(a)
                self.add_frames_num('inst', a)

        self.samples = dict(self.samples)

    # def mixtures(self):
    #     items = list(self.samples.values())
    #     random.shuffle(items)
    #     for item in items:
    #         if item['mix'] and item['vocl']:
    #             yield item['mix'], item['vocl']

    def get_mixture_with_vocal(self, filename):
        key = os.path.basename(filename)[:-6]
        if key not in self.samples:
            print('[+] Musdb18::get_mixture_with_vocal:: key was not found in samples', key)
            return None
        val = self.samples[key]
        if 'mix' not in val:
            print('[+] Musdb18::get_mixture_with_vocal:: mix was not found in val', val.keys())
            return None

        return val['mix']
    
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


class Quasi(Dataset):
    VOCAL_KEYWORDS = ['choir', 'speech', 'lv_', 'harmo', 'vox', 'voix', 'voic', 'voc']
    
    def __init__(self, params):
        super().__init__()
        self.samples = defaultdict(lambda: {'mix': [], 'vocl': [], 'inst': []})
        path = params['path']
        self.net_vocals = 0
        self.net_insts = 0
        for a in iterate_files(os.path.join(path, 'separation'), '.wav'):
            key = os.path.basename(os.path.dirname(os.path.dirname(a))).lower()
            filename = os.path.basename(a).lower()
            if self._is_vocal_name(filename):
                self.samples[key]['vocl'].append(a)
                # self.net_vocls += get_net_duration(a)
                self.add_frames_num('vocl', a)
                continue
            if 'mix' in filename:
                self.samples[key]['mix'].append(a)
                continue

            self.samples[key]['inst'].append(a)
            self.add_frames_num('inst', a)
            # self.net_insts += get_net_duration(a)
        self.samples = dict(self.samples)

    def get_mixture_with_vocal(self, filename):
        key = os.path.basename(os.path.dirname(os.path.dirname(filename))).lower()
        if key not in self.samples:
            print('[+] Quasi::get_mixture_with_vocal:: key was not found in samples', key)
            return None
        val = self.samples[key]
        if 'mix' not in val:
            print('[+] Quasi::get_mixture_with_vocal:: mix was not found in val', val.keys())
            return None

        return val['mix']

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
        self.vocl_frames = 0
        self.inst_frames = 0
        for ds_config in params:
            self._load_dataset_with_config(ds_config)
        print('[+] {} datasets were loaded'.format(len(self.datasets)))

    def _load_dataset_with_config(self, config):
        print('[+] MultiDatasets::_load_dataset_with_config:', config)
        cache = config.get('cache')
        if cache and os.path.isfile(cache):
            print('[+] getting from cache!', cache)
            ds = pickle.load(open(cache, 'rb'))
            self.datasets.append(ds)
            self.vocl_frames += ds.get_frames_num('vocl')
            self.inst_frames += ds.get_frames_num('inst')
            return
        
        _cls = globals()[config['type']]
        ds = _cls(config['params'])
        self.datasets.append(ds)
        self.vocl_frames += ds.get_frames_num('vocl')
        self.inst_frames += ds.get_frames_num('inst')

        if cache:
            pickle.dump(ds, open(cache, 'wb'))

    def vocals(self):
        return self._iterate_iterators(self.vocl_frames, label_name='vocl')
    
    def instrumentals(self):
        return self._iterate_iterators(self.inst_frames, label_name='inst')

    def _iterate_iterators(self, perm_size, label_name):        
        while True:
            print('[+] generating permutations:', perm_size, label_name)
            # perm = np.random.permutation(perm_size)
            # perm = range(perm_size)
            print('[+] permutations generated:', perm_size, label_name)
            # for offset in perm:
            while True:
                try:
                    # print('[+] next', label_name)
                    offset = random.randint(0, perm_size-1)
                    # open('offsets.log', 'a').write(str(offset) + '\n')
                    # print('[+] offset', offset, label_name)
                    for ds in self.datasets:
                        # print('[+] offset', offset, label_name)
                        # print('[+] ds', ds, label_name)
                        cur_frames = ds.get_frames_num(label_name)
                        # print('[+] cur_frames', cur_frames, label_name)
                        if offset <= cur_frames:
                            # print('[+] in!', label_name)
                            if label_name == 'inst':
                                # print('[+] inst!')
                                ret = (ds.get_with_perm(label_name, offset), [1, 0])
                            else:
                                # print('[+] voc!')
                                ret = (ds.get_with_perm(label_name, offset), [0, 1])
                                params, label = ret
                                filename, offset, _range = params
                                mix = ds.get_mixture_with_vocal(filename)
                                # print('data:723:', mix)
                                if mix:
                                    if isinstance(mix, list):
                                        mix = mix[0]
                                    # print('[+] hasmix!')
                                    ret = MixWithVocalResult(ret, mix)
                            # print('[+] yielding', label_name, ret)
                            yield ret
                            break
                        # print('[+] continue to next!', label_name)
                        
                        offset -= cur_frames
                    # print('[+] out of loop!', label_name)
                except:
                    traceback.print_exc()
                    raise
                # print('[+] going for next one', label_name)
            # print('[+] finished perms for:', label_name)
    # def _iterate_iterators(self, iter_func, label):
    #     while True:
    #         iters = iter_func()
    #         while iters:
    #             random.shuffle(iters)
    #             ended = []
    #             # print('[+] iterators:', iters)
                
    #             for i, iterator in enumerate(iters):
    #                 try:
    #                     # print('[+] iterating', i, iterator)
    #                     x = next(iterator)
    #                     # print('[+] iterating', x, label)
    #                     yield x, label
    #                 except StopIteration:
    #                     ended.append(i)
                
    #             for i in reversed(sorted(ended)):
    #                 del iters[i]
    #                 print('[+] iterator has finished for', i, label)
            
    #         print('[+] finished all iterators')
                

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