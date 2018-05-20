import os
import sys
import json
import librosa
import pickle
import numpy as np
from scipy.ndimage.interpolation import shift
import audio

sys.path.insert(0,'..')

AUDIO_EXTS = [b'.wav', b'.mp3', b'.ogg', b'.aac', b'.wma']


SMOOTHING_LEVEL = 80
def smooth(arr, level):
    res = arr.copy()
    for i in range(level):
        res += shift(arr, i+1)
        res += shift(arr, -(i+1))
    # import pdb; pdb.set_trace()
    return (res >= (level + 1)).astype(np.float32)
    # return res / (level + 1)

def get_loud_mask(im, _max):
    # print(im.shape)
    thresh = 0.030
    # print('mean: {0:.4f} max: {0:.4f}'.format(im.mean(), _max))
    ret = ((im / _max) > (thresh)).astype(np.float32)
    return smooth(ret, SMOOTHING_LEVEL)


def get_silence_mask(im, _max):
    # print(im.shape)
    thresh = 0.00055
    # print('mean: {0:.4f} max: {0:.4f}'.format(im.mean(), _max))
    ret = ((im / _max) < (thresh)).astype(np.float32)
    return smooth(ret, SMOOTHING_LEVEL)


def get_ranges_from_mask(mask):
    coords = np.where((mask + shift(mask, 1)) == 1.0)[0].tolist()
    if mask[-1] == 1.0:
        coords.append(len(mask)-1)
    return coords


def get_audio(filename):
    print(filename)
    y, sr = librosa.load(filename,
                    sr=audio.SAMPLE_RATE,
                    mono=False)

    if y.ndim > 1:
        y = y[0]
    return y

def get_spect(y):
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=audio.SAMPLE_RATE,
                                         n_mels=audio.MELS,
                                         n_fft=audio.FFT,
                                         power=audio.POWER,
                                         hop_length=audio.HOP_LENGTH)
    graph = librosa.power_to_db(mel, ref=np.max)
    graph = graph.clip(-80, 0) + 80 
    graph = graph / 80

    return graph

def get_ranges(y):
    rms = librosa.feature.rmse(y=y).flatten()

    loudness_mask = get_loud_mask(rms, _max=rms.max())
    loudness_ranges = get_ranges_from_mask(loudness_mask)

    silence_mask = get_silence_mask(rms, _max=rms.max())
    silence_mask_ranges = get_ranges_from_mask(silence_mask)
    
    loudness_ranges = list(zip(loudness_ranges[0::2], loudness_ranges[1::2]))
    silence_mask_ranges = list(zip(silence_mask_ranges[0::2], silence_mask_ranges[1::2]))

    return loudness_ranges, silence_mask_ranges

def main():
    print(sys.argv[1:])
    if len(sys.argv) < 2:
        print('Usage: python gen_silence_map.py <audiofile>')
        return
    
    filename = [a for a in sys.argv[1:] if os.path.isfile(a)][0].encode("utf-8")

    pid = os.getpid()
    print()
    print('[+][{:>6}] is Working on {}'.format(pid, filename))
    base, ext = os.path.splitext(filename)
    if ext not in AUDIO_EXTS:
        print('[!][{:>6}] This is not an audio file!'.format(pid))
        raise RuntimeError('unknown ext', ext)

    # print('[+][{:>6}] getting info..'.format(pid))
    # info = {}
    # fileinfo_path = base + b'.json'
    # if os.path.isfile(fileinfo_path):
    #     try:
    #         info = json.loads(open(fileinfo_path, 'r').read())
    #     except:
    #         print('[!][{:>6}] Error during info load..'.format(pid))

    # if 'loudness_ranges' in info:
    #     if '--force' not in sys.argv:
    #         print('[!][{:>6}] has ranges.. skipping..'.format(pid))
    #         return
    #     print('[+][{:>6}] running over existing info..'.format(pid))


    spectrum_file = base + b'.mel'
    y = get_audio(filename.decode("utf-8"))
    spect = get_spect(y).astype(np.float32)
    open(spectrum_file.decode("utf-8"), 'wb').write(spect.T.tobytes())
    return
    loudness_ranges, silence_mask_ranges = get_ranges(y)
    
            
    
    # print('[+][{:>6}] Loading from audio file..'.format(pid))
    # sound = pydub.AudioSegment.from_file(filename)
    # print('[+][{:>6}] Applying gain: {}'.format(pid, -sound.max_dBFS))
    # sound = sound.apply_gain(-sound.max_dBFS)
    # print('[+][{:>6}] detecting silence...'.format(pid))
    # silence = pydub.silence.detect_silence(sound, min_silence_len=1000, silence_thresh=-35)
    if 'silence' in info:
        del info['silence']
    
    info['loudness_ranges'] = loudness_ranges
    info['silence_mask_ranges'] = silence_mask_ranges
    print('[+][{:>6}] writing info to disk: {}'.format(pid, fileinfo_path))
    open(fileinfo_path.decode("utf-8"), 'w').write(json.dumps(info))
    
    



if __name__ == '__main__':
    main()


