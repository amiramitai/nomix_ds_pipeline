
import math
import random
import os
import pickle
import datetime
import json

import audioread
import librosa
import numpy as np

from exceptions import BadAudioFile


FFT = 2048
HOP_LENGTH = int(FFT / 4)
MELS = 224
SAMPLE_RATE = 44100
POWER = 2.0
SPLIT_HOP_LENGTH = int(MELS / 8)  # Cols
SILENCE_HOP_THRESHOLD = 0.30  # 30%
RMS_SILENCE_THRESHOLD = 0.75  # RMS
SECONDS_TO_READ = 2.8
SAMPLE_MIN_LENGTH = int(MELS * 0.6)


def get_number_of_frames(filename):
    try:
        n = 0
        loudness_ranges = get_loudness_ranges(filename)
        for start, end in loudness_ranges:
            if (end - start) < SAMPLE_MIN_LENGTH:
                continue

            net_frames = end - start - SAMPLE_MIN_LENGTH
            n += net_frames
        return n
    except RuntimeError:
        return 0


def get_net_duration(filename):
    dur = get_duration(filename)
    metadata_filename = os.path.splitext(filename)[0] + '.json'
    if not os.path.isfile(metadata_filename):
        raise RuntimeError('Metadata is not there for file', filename)
    j = json.loads(open(metadata_filename).read())
    for start_ms, end_ms in j.get('silence', []):
        dur -= (end_ms - start_ms) / 1000.0
        
    return dur


def get_duration(filename):
    try:
        with audioread.audio_open(filename) as f:
            return f.duration
    except:
        print(filename)
        raise
    raise RuntimeError('Something went wrong', self.filename)


class AudioFile:
    def __init__(self, filename):
        self.filename = filename
        self.duration = None
        self.cursor = 0

    def get_duration(self):
        if not self.duration:
            self.duration = get_duration(self.filename)
        return self.duration

    def seek(self, offset):
        self.cursor = offset

    def read(self, duration):
        y, sr = librosa.load(self.filename,
                             sr=SAMPLE_RATE,
                             mono=False,
                             offset=self.cursor,
                             duration=duration)
        return y


def format_secs(secs):
    mins = int(secs / 60.0)
    secs -= mins * 60
    return '{}:{}'.format(mins, secs)


class AudioPatch:
    def __init__(self, data, filename, loc):
        self.filename = filename
        self.loc = loc
        self.data = data
        self.min = str(int(datetime.datetime.now().minute / 3))


def get_non_silent_range(filename):
    dur = get_duration(filename)
    jfilename = os.path.splitext(filename)[0] + '.json'
    j = json.load(open(jfilename))
    silence_ranges = j['silence']
    start = 0
    end = dur
    non_silent_ranges = []
    last = 0
    for i, _range in enumerate(silence_ranges):
        if i == 0 and _range[0] == 0:
            last = _range[1]
            continue

        non_silent_ranges.append([last, _range[0]])
        last = _range[1]

    if non_silent_ranges:
        return random.choice(non_silent_ranges)
    return [0, dur]


def pick_within_range(start, end):
    # if it's more then we need, pick a random start position
    if (end - start) > MELS:
        pos = random.randint(start, end-MELS)
        return pos, pos+MELS

    # it's just between the minimum requirement and max requirement
    return start, end


def choose_spect_range(ranges):
    while ranges:
        rindex = random.randint(0, len(ranges) - 1)
        start, end = ranges[rindex]

        # if it's below minimum acceptable length, drop it and continue
        if (end - start) < SAMPLE_MIN_LENGTH:
            ranges.pop(rindex)
            continue

        return pick_within_range(start, end)


def get_loudness_ranges(audio_filename):
    jfilename = os.path.splitext(audio_filename)[0] + '.json'
    j = json.load(open(jfilename))
    if 'loudness_ranges' not in j:
        print(audio_filename, jfilename)
    return j['loudness_ranges']


def get_rand_loudness_range(audio_filename):
    jfilename = os.path.splitext(audio_filename)[0] + '.json'
    j = json.load(open(jfilename))
    if 'loudness_ranges' not in j:
        print(audio_filename, jfilename)
    loudness_ranges = j['loudness_ranges'][:]

    if not loudness_ranges:
        raise RuntimeError('loudness_ranges are missing', audio_filename)

    ret = choose_spect_range(loudness_ranges)
    if not ret:
        raise RuntimeError('no suitable ranges were found', audio_filename, j['loudness_ranges'])

    return ret


def get_mel_filename(audio_filename):
    return os.path.splitext(audio_filename)[0] + '.mel'


def get_mel_spect(mel_filename, _range, dtype=np.float32):
    with open(mel_filename, 'rb') as f:
        itemsize = np.dtype(dtype).itemsize
        start, end = _range
        f.seek(MELS * start * itemsize)
        spect = np.frombuffer(f.read((end - start) * MELS * itemsize), dtype=dtype)
        spect = spect.reshape((-1, MELS)).T
        return spect


def get_spect_range_from_time_range(time_range):
    # print('[+] getting spect range from time range')
    # column_length = HOP_LENGTH / SAMPLE_RATE
    start, end = time_range
    start = int(np.round(start / (HOP_LENGTH / SAMPLE_RATE)))
    end = int(np.round(end / (HOP_LENGTH / SAMPLE_RATE)))

    return start, end


def get_range_with_offset_and_ranges(offset, ranges):
    for start, end in ranges:
        avail_frames = (end - SAMPLE_MIN_LENGTH + 1) - start
        if avail_frames <= 0:
            continue

        if offset > avail_frames:
            offset -= avail_frames
            continue

        nstart = start + offset
        if end - nstart > MELS:
            end = nstart + MELS
        return (nstart, end)


def get_loudness_range_with_offest(audio_filename, offset):
    loudness_ranges = get_loudness_ranges(audio_filename)
    return get_range_with_offset_and_ranges(offset, loudness_ranges)


def get_offset_range_patch(audio_filename, offset, _range=None, mix_filename=None):
    path = 1
    if _range is None:
        _range = get_loudness_range_with_offest(audio_filename, offset)
    elif len(_range) == 2:
        path = 2
        start, end = _range
        start += offset
        if end - start > MELS:
            end = start + MELS
        _range = (start, end)
    else:
        path = 3
        _range = get_range_with_offset_and_ranges(offset, _range)

    if mix_filename:
        mel_filename = get_mel_filename(mix_filename)
    else:
        mel_filename = get_mel_filename(audio_filename)
    # print(mel_filename, _range)
    res = get_mel_spect(mel_filename, _range)
    # res = mel_spect.T[_range[0]:_range[1]].T
    if res.shape == (MELS, MELS):
        return res

    # print('[+] extending patch!', res.shape, _range, audio_filename)
    start, end = _range
    if (end - start) < SAMPLE_MIN_LENGTH:
        raise RuntimeError("Too short", audio_filename, offset, _range, mix_filename, path)

    patch = np.zeros((MELS, MELS))
    patch[0:res.shape[0], 0:res.shape[1]] = res
    return patch


def get_rand_spect_patch(audio_filename, _range=None):
    if _range:
        # print('[+] pick_within')
        _range = pick_within_range(*_range)
    else:
        # print('[+] get_rand_loudness_range')
        _range = get_rand_loudness_range(audio_filename)

    mel_filename = get_mel_filename(audio_filename)
    res = get_mel_spect(mel_filename, _range)
    # res = mel_spect.T[_range[0]:_range[1]].T
    if res.shape == (MELS, MELS):
        return res

    # print('[+] extending patch!', res.shape, _range, audio_filename)
    start, end = _range
    if (end - start) < SAMPLE_MIN_LENGTH:
        raise RuntimeError("Too short", audio_filename, _range)

    patch = np.zeros((MELS, MELS))
    patch[0:res.shape[0], 0:res.shape[1]] = res
    return patch


def get_rand_audio_patch(filename, _range=None):
    if not _range:
        _range = get_non_silent_range(filename)
    
    CPRECISION = 0.01
    toread = math.ceil(((MELS * HOP_LENGTH) / SAMPLE_RATE) / CPRECISION) * CPRECISION
    sample_loc = random.uniform(_range[0], _range[1] - toread)
    y, sr = librosa.load(filename,
                         sr=SAMPLE_RATE,
                         mono=False,
                         offset=sample_loc/1000.0,
                         duration=toread)
    if y.ndim > 1:
        y = random.choice(y)
    return y

    CPRECISION = 0.01
    toread = math.ceil(((MELS * HOP_LENGTH) / SAMPLE_RATE) / CPRECISION) * CPRECISION
    for i in range(10):
        loc = random.uniform(0, dur-toread)
        af.seek(loc)
        y = af.read(toread)
        if y.ndim == 2:
            y = y[0]

        if len(y) < MELS * HOP_LENGTH:
            print('[+] get_audio_patch_with_params: got too short for {}/y:{}/loc:{}/dur:{}'.format(filename, len(y), format_secs(loc), dur))
            continue
        rms = librosa.feature.rmse(y=y) 
        # if rms.mean() > 0.05:
        #     return y
        if sum(rms[0]) > 5.0:
            # print(len(y), filename)
            return AudioPatch(y, filename, loc)
    
    raise BadAudioFile()


def get_image_with_audio(y, label):
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=SAMPLE_RATE,
                                         n_mels=MELS,
                                         n_fft=FFT,
                                         power=POWER,
                                         hop_length=HOP_LENGTH)
    
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = mel_db.T[0:MELS]
    image = (image.clip(-80, 0) + 80) / 80
    image.reshape((MELS, MELS))
    return image, label


def to_audiosegment(arr):
    if arr.dtype in [np.float16, np.float32, np.float64]:
        arr = np.int16(arr/np.max(np.abs(arr)) * 32767)
    
    return AudioSegment(arr.tobytes(),
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import traceback
    fig = plt.figure(figsize=(16,4))
    a=fig.add_subplot(1,1,1)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # fig.axes[0].set_visible(False)
    # fig.axes[1].set_visible(False)
    import pickle
    jaud = pickle.load(open(r"T:\cache\jamaudio.pickle", 'rb'))
    for i, (filename, _range) in enumerate(jaud.instrumentals()):
        print(i)
        # filename = r'T:\datasets\quasi\separation\set1\vieux_farka-ana\Bal_Pan_Eq_Comp_Fx\vox1.wav'
        try:
            patch = get_rand_spect_patch(filename, _range)
            print(patch.shape)
        except:
            traceback.print_exc()
        # plt.imshow(patch, cmap='hot')
        # plt.show()