
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



    # if _range == len(silence):

    # print('[+] {} getting audio patch'.format(i))
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



# def get_audio_patch_with_params(filename):
#     af = AudioFile(filename)
#     dur = af.get_duration()
#     # print('[+] {} getting audio patch'.format(i))
#     CPRECISION = 0.01
#     toread = math.ceil(((MELS * HOP_LENGTH) / SAMPLE_RATE) / CPRECISION) * CPRECISION
#     for i in range(10):
#         loc = random.uniform(0, dur-toread)
#         af.seek(loc)        
#         y = af.read(toread)
#         if y.ndim == 2:
#             y = y[0]


#         if len(y) < MELS * HOP_LENGTH:
#             print('[+] get_audio_patch_with_params: got too short for {}/y:{}/loc:{}/dur:{}'.format(filename, len(y), format_secs(loc), dur))
#             continue
#         rms = librosa.feature.rmse(y=y) 
#         # if rms.mean() > 0.05:
#         #     return y
#         if sum(rms[0]) > 5.0:
#             # print(len(y), filename)
#             return AudioPatch(y, filename, loc)
    
#     raise BadAudioFile()
        # else:
        #     fmtstr = '[+] {} patch is not loud enough:\n\t{}\n\t{}\n\t{}\n\t{}'
        #     print(fmt.str.format(i, filename, format_secs(loc), rms.mean(), sum(rms[0])))


def get_image_with_audio(y, label):
    # print('[+] {} turning to mel..'.format(i))
    # print(len(y))
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=SAMPLE_RATE,
                                         n_mels=MELS,
                                         n_fft=FFT,
                                         power=POWER,
                                         hop_length=HOP_LENGTH)
    
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = mel_db.T[0:MELS].T
    image = (image.clip(-80, 0) + 80) / 80
    return image, label


def to_audiosegment(arr):
    if arr.dtype in [np.float16, np.float32, np.float64]:
        arr = np.int16(arr/np.max(np.abs(arr)) * 32767)
    
    return AudioSegment(arr.tobytes(),
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=1)
