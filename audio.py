
import math
import random
import os
import pickle
import datetime
import json
import sys

import audioread
import librosa
import numpy as np

from exceptions import BadAudioFile


FFT = 2048
ROWS = (FFT // 2) + 1
HOP_LENGTH = FFT // 4
# MELS = 224
# MELS = 1024
FRAMES = 224
SAMPLE_RATE = 44100
POWER = 2.0
SILENCE_HOP_THRESHOLD = 0.30  # 30%
RMS_SILENCE_THRESHOLD = 0.75  # RMS
SECONDS_TO_READ = 2.8
SAMPLE_MIN_LENGTH = 60


def get_number_of_frames(filename):
    try:
        n = 0
        loudness_rangets = get_loudness_ranges(filename)
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
    silence_ranges = j['silence_mask_ranges']
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
    if (end - start) > FRAMES:
        pos = random.randint(start, end-FRAMES)
        return pos, pos+FRAMES

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
        print('[+] loudness_ranges was not found in', audio_filename, jfilename)
    return j['loudness_ranges']


def get_rand_loudness_range(audio_filename):
    jfilename = os.path.splitext(audio_filename)[0] + '.json'
    j = json.load(open(jfilename))
    if 'loudness_ranges' not in j:
        print('[+] loudness_ranges was not found in', audio_filename, jfilename)
    loudness_ranges = j['loudness_ranges'][:]

    if not loudness_ranges:
        raise RuntimeError('loudness_ranges are missing', audio_filename)

    ret = choose_spect_range(loudness_ranges)
    if not ret:
        raise RuntimeError('no suitable ranges were found', audio_filename, j['loudness_ranges'])

    return ret


def get_mel_filename(audio_filename):
    if MELS == 224:
        return os.path.splitext(audio_filename)[0] + '.mel'

    return '{}.{}.mel'.format(os.path.splitext(audio_filename)[0], MELS)


def audio_path_to_cache_path(path):
    if sys.platform == 'darwin':
        return os.path.join('/Users/amiramitai/cache', '{}.{}.fft'.format(os.path.basename(path), FFT))

    t = 'T{}.{}.fft'.format(path[1:], FFT)
    if os.path.isfile(t):
        return t
    v = 'V{}.{}.fft'.format(path[1:], FFT)
    if os.path.isfile(v):
        return v
    s = 'S{}.{}.fft'.format(path[1:], FFT)
    return s


def get_spect_from_cache(spect_filename, _range, dtype=np.float32):
    with open(spect_filename, 'rb') as f:
        itemsize = np.dtype(dtype).itemsize
        start, end = _range
        f.seek(ROWS * start * itemsize)
        spect = np.frombuffer(f.read((end - start) * ROWS * itemsize), dtype=dtype)
        spect = spect.reshape((-1, ROWS)).T
        return spect


def get_spect(audio_filename, _range):
    # get the spect
    # print('[+] getting audio patch')
    # import time
    # start = time.time()
    cache_path = audio_path_to_cache_path(audio_filename)
    if cache_path and os.path.isfile(cache_path):
        return get_spect_from_cache(cache_path, _range)
    
    # raise RuntimeError('now we are running on cache only', cache_path, audio_filename, _range)
    
    patch = get_audio_patch(audio_filename, _range)
    spect = get_image_with_audio(patch)
    return spect[:, :FRAMES]


def get_mel_spect(audio_filename, _range, dtype=np.float32):
    mel_filename = get_mel_filename(audio_filename)

    if not os.path.isfile(mel_filename):
        # get the spect
        # print('[+] getting audio patch')
        patch = get_audio_patch(audio_filename, _range)
        # print('[+] get image with audio')
        spect = get_image_with_audio(patch)
        # print('[+] done')
        return spect

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
        if end - nstart > FRAMES:
            end = nstart + FRAMES
        return (nstart, end)


def get_loudness_range_with_offest(audio_filename, offset):
    loudness_ranges = get_loudness_ranges(audio_filename)
    ret = get_range_with_offset_and_ranges(offset, loudness_ranges)
    # print('get_loudness_rwo: {} offset: {} lranges: {} range: {}'.format(audio_filename, offset, loudness_ranges, ret))
    return ret


def get_offset_range_patch(audio_filename, offset, _range=None, mix_filename=None, rows=ROWS, cols=FRAMES):
    path = 1
    if _range is None:
        _range = get_loudness_range_with_offest(audio_filename, offset)
    elif len(_range) == 2:
        path = 2
        start, end = _range
        try:
            start += offset
        except:
            print('[!] audio.py: range except:', _range)
            raise
        if end - start > cols:
            end = start + cols
        _range = (start, end)
    else:
        path = 3
        _range = get_range_with_offset_and_ranges(offset, _range)

    filename = mix_filename
    if not filename:
        filename = audio_filename
    # print(mel_filename, _range)
    # print(_range, path)
    # res = get_mel_spect(filename, _range)

    res = get_spect(filename, _range)
    # res = mel_spect.T[_range[0]:_range[1]].T
    if res.shape == (rows, cols):
        return res

    # print('[+] extending patch!', res.shape, _range, audio_filename)
    start, end = _range
    if (end - start) < SAMPLE_MIN_LENGTH:
        raise RuntimeError("Too short", audio_filename, offset, _range, mix_filename, path)

    patch = np.zeros((rows, cols))
    patch[0:res.shape[0], 0:res.shape[1]] = res
    return patch


def get_audio_patch(filename, _range=None):
    if _range is None:
        _range = get_non_silent_range(filename)
    
    CPRECISION = 0.01
    toread = math.ceil(((FRAMES * HOP_LENGTH) / SAMPLE_RATE) / CPRECISION) * CPRECISION
    sample_loc = random.uniform(_range[0], _range[1] - toread)
    y, sr = librosa.load(filename,
                         sr=SAMPLE_RATE,
                         mono=False,
                         offset=sample_loc/1000.0,
                         duration=toread)
    if y.ndim > 1:
        # y = random.choice(y)
        y = y[0]
    return y


def get_image_with_audio(y):
    # mel = librosa.feature.melspectrogram(y=y,
    #                                      sr=SAMPLE_RATE,
    #                                      n_mels=MELS,
    #                                      n_fft=FFT,
    #                                      power=POWER,
    #                                      hop_length=HOP_LENGTH)
    S = librosa.stft(y, FFT, HOP_LENGTH)
    # mag, phase = librosa.magphase(S)
    # print(mag.min(), mag.max(), mag.mean())
    win_len = FFT
    # max_val = (win_len / 3.0)
    max_val = (win_len / 3.0)
    # return mag / max_val
    power = librosa.power_to_db(S ** 2, ref=np.max)
    return (power.clip(-80, 0) + 80) / 80


def spectrum_to_mel_spectrum(S, n_mels):
    mel_basis = librosa.filters.mel(SAMPLE_RATE, FFT, n_mels=n_mels)
    return np.dot(S.T, mel_basis.T).T


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
    fig = plt.figure(figsize=(16, 4))
    a = fig.add_subplot(1, 1, 1)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # fig.axes[0].set_visible(False)
    # fig.axes[1].set_visible(False)
    import pickle
    jaud = pickle.load(open(r"T:\cache\jamaudio.pickle", 'rb'))