
import math
import random
import os
import pickle
import datetime

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


class AudioFile:
    def __init__(self, filename):
        self.filename = filename
        self.duration = None
        self.cursor = 0

    def get_duration(self):
        if not self.duration:
            with audioread.audio_open(self.filename) as f:
                self.duration = f.duration
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

    def cache(self, cls):
        dir_path = os.path.join(os.environ['HOME'], 'cache', str(cls), self.min)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        file_name = '{}_{}'.format(os.path.basename(self.filename), str(self.loc))
        open(os.path.join(dir_path, file_name), 'wb').write(pickle.dumps(self.data))



def get_audio_patch_with_params(filename):
    af = AudioFile(filename)
    dur = af.get_duration()
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
