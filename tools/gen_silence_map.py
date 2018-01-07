import os
import sys
import json


import pydub

AUDIO_EXTS = ['.wav', '.mp3', '.ogg', '.aac', '.wma']

def main():
    if len(sys.argv) < 2:
        print('Usage: python gen_silence_map.py <audiofile>')
        return

    filename = [a for a in sys.argv[1:] if os.path.isfile(a)][0]

    pid = os.getpid()
    print('[+][{:>6}] is Working on {}'.format(pid, filename))
    base, ext = os.path.splitext(filename)
    if ext not in AUDIO_EXTS:
        print('[!][{:>6}] This is not an audio file!'.format(pid))
        raise RuntimeError('unknown ext', ext)

    print('[+][{:>6}] getting info..'.format(pid))
    info = {}
    fileinfo_path = base + '.json'
    if os.path.isfile(fileinfo_path):
        try:
            info = json.loads(open(fileinfo_path, 'r').read())
        except:
            print('[!][{:>6}] Error during info load..'.format(pid))

    if 'silence' in info:
        if '--force' not in sys.argv:
            print('[!][{:>6}] Silence is already in info.. skipping..'.format(pid))
            return
        print('[+][{:>6}] running over existing info..'.format(pid))

    
    print('[+][{:>6}] Loading from audio file..'.format(pid))
    sound = pydub.AudioSegment.from_file(filename)
    print('[+][{:>6}] Applying gain: {}'.format(pid, -sound.max_dBFS))
    sound = sound.apply_gain(-sound.max_dBFS)
    print('[+][{:>6}] detecting silence...'.format(pid))
    silence = pydub.silence.detect_silence(sound, min_silence_len=1000, silence_thresh=-35)
    info['silence'] = silence
    print('[+][{:>6}] writing info to disk: {}'.format(pid, fileinfo_path))
    open(fileinfo_path, 'w').write(json.dumps(info))
    
    



if __name__ == '__main__':
    main()


