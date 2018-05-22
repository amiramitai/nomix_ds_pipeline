import numpy as np
import librosa
import audio
import matplotlib.pyplot as plt
import os
import time
import logging
import sys
import json
import pickle


from functools import partial
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.getLogger("tensorflow").setLevel(logging.WARNING)

filename = '/Users/amiramitai/Projects/nomix/02 Made In Heaven.mp3'
# filename = '/Users/amiramitai/Projects/nomix/Cool_Joke_-_Undo_OST_Fullmetal_Alchemist_OP_3_(DemoLat.com).mp3'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-06_085120'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-06_104506'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-06_171143'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-09_064212'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-09_214438'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-10_171313'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-10_221942'
# weights = '/Users/amiramitai/Projects/nomix/rnn2-2018-05-14_115504'
# weights = '/Users/amiramitai/Projects/nomix/1-2018-05-16_201203'
# weights = '/Users/amiramitai/Projects/nomix/1-2018-05-17_075141'
# weights = '/Users/amiramitai/Projects/nomix/1-2018-05-17_184222'
# weights = '/Users/amiramitai/Projects/nomix/1-2018-05-18_185354'
# weights = '/Users/amiramitai/Projects/nomix/1-2018-05-19_145340'
# weights = '/Users/amiramitai/Projects/nomix/0-2018-05-19_171224'
# weights = '/Users/amiramitai/Projects/nomix/1-2018-05-19_210748'
weights = '/Users/amiramitai/Projects/nomix/0-2018-05-22_090306'


try:
    state = json.load(open(os.path.join(weights, 'state.json')))
    params = state['params']
except:
    print('fallback')
    params = dict(hidden_size=1024, seq_len=12)


def my_get_latest(weights):
    lc = tf.train.latest_checkpoint(weights)
    if lc:
        return lc
    cp = os.path.join(weights, 'checkpoint')
    model = open(cp, 'r').read().split('\n')[0].split('\\\\')[-1][:-1]
    return os.path.join(weights, model)
    ncp = 'model_checkpoint_path: "{}"\r\nall_model_checkpoint_paths: "{}"'.format(model, model)
    open(cp, 'w').write(ncp)
    return tf.train.latest_checkpoint(weights)


def split_spectrogram_into_snippets(spectrogram, length, zero_padding=False):
    snippets = []
    spectrogram_length = spectrogram.shape[1]

    if zero_padding and (spectrogram_length % length != 0):

        # Do zero padding (this is relatively expensive)
        new_length = int(ceil(spectrogram_length * 1. / length)) * length
        new_spectrogram = np.zeros((spectrogram.shape[0], new_length))
        new_spectrogram[:, :spectrogram_length] = spectrogram
        spectrogram = new_spectrogram
        length = new_length

    # Create now the snippets
    snippet_count = int(spectrogram_length / length)

    # Create all snippets
    for i in range(snippet_count):
        snippets.append(spectrogram[:, (i * length):((i + 1)*length)])

    return snippets

win_length = audio.FFT
print('[+] loading audio file:', filename)
try:
    S = pickle.load(open(filename + '.cache', 'rb'))
except:
    print('recaching..')
    y, sr = librosa.load(filename, sr=audio.SAMPLE_RATE)
    if y.ndim > 1:
        y = y[0]
    S = librosa.stft(y, audio.FFT, audio.HOP_LENGTH)
    # mag, phase = librosa.magphase(S)
    pickle.dump(S, open(filename + '.cache', 'wb'))

# mel spectrum
# mel_basis = librosa.filters.mel(audio.SAMPLE_RATE, audio.FFT, n_mels=audio.MELS)
# mel = np.dot(mel_basis, mag)
# print(mel.shape, mag.shape)

# mel_graph = librosa.power_to_db(mel ** 2.0, ref=np.max)
# mel_graph = (mel_graph.clip(-80, 0) + 80) / 80.0
# print('mel_graph.shape:', mel_graph.shape)
mag, phase = librosa.magphase(S)
power = librosa.power_to_db(S, ref=np.max)
spect = (power.clip(-80, 0) + 80) / 80

# plt.imshow(feature, cmap='Greys_r')
# plt.show()
SEQ_LEN = params['seq_len']
IM_HEIGHT = 224
IM_WIDTH = 224
IM_CHANNEL = 1
models_to_keep = 3
models_history = []
checkpoint_steps = 250
model_version = 1
K = IM_WIDTH * 64
num_classes = 2

weights_name = '.' + weights.split('_')[-1]
l0_out_filename = '{}.l0.{}.mask'.format(filename, weights_name)
l1_out_filename = '{}.l1.{}.mask'.format(filename, weights_name)
logits0_snips = np.zeros(spect.shape)
logits1_snips = np.zeros(spect.shape)
extract_mask = True
try:
    logits0_snips = pickle.load(open(l0_out_filename, 'rb'))
    logits1_snips = pickle.load(open(l1_out_filename, 'rb'))
    extract_mask = False
except:
    extract_mask = True

if extract_mask:
    import tensorflow as tf
    from rnn import RNNModel
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    with tf.name_scope('tower_0') as scope:
        pass

    x_mixed = tf.placeholder(tf.float32, shape=(None, None, audio.ROWS), name='x_mixed')
    y_src1 = tf.placeholder(tf.float32, shape=(None, None, audio.ROWS), name='y_src1')
    y_src2 = tf.placeholder(tf.float32, shape=(None, None, audio.ROWS), name='y_src2')
    global_step = tf.Variable(0, trainable=False)
    net = RNNModel(x_mixed, y_src1, y_src2, params)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    config = tf.ConfigProto(log_device_placement=True)
    # config = tf.ConfigProto(device_count={'GPU': 0})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)

    to_load = []
    for v in tf.trainable_variables():
        to_load.append(v)
    s = tf.train.Saver(to_load, max_to_keep=None)
    latest = my_get_latest(weights)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(latest)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print('-----------------------------')
    print('-----------------------------')
    print('-----------------------------')
    print('-----------------------------')
    ops = []
    for v in tf.global_variables():
        name = v.name

        name = name.split(':')[0]
        if name in var_to_shape_map:
            print('[+] assigning:', v.name, np.all(np.isfinite(reader.get_tensor(name))))
            ops.append(tf.assign(v, reader.get_tensor(name)))
            continue

        orig_name = name
        name = name[4:]
        if name in var_to_shape_map:
            print('[+] assigning:', v.name, np.all(np.isfinite(reader.get_tensor(name))))
            ops.append(tf.assign(v, reader.get_tensor(name)))
            continue

        print('[+] skipping:', orig_name)
        
    sess.run(ops)
    # for k, v in var_to_shape_map.items():
    #     if 'Adam' in k:
    #         continue
    #     print(k)
    # import sys
    # sys.exit(1)
    # latest = tf.train.latest_checkpoint(weights)
    # s.restore(sess, latest)

    # snippets = split_spectrogram_into_snippets(mel_graph, 224)

    preds_snips = []
    # logits0_snips = []
    # logits1_snips = []
    to_run = [net.prediction]

    cmap = 'hot'
    # cmap = 'Greys_r'
    # for i, s in enumerate(snippets):

    import datetime
    start = datetime.datetime.now()
    batch_size = 16
    # for i in range(steps):
    i = 0

    def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # Padding
        pad_len = 0
        if n_frames % SEQ_LEN > 0:
            pad_len = (SEQ_LEN - (n_frames % SEQ_LEN))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % SEQ_LEN == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, SEQ_LEN, freq))
        return batch


    def batch_to_spec(src, num_wav):
        # shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq, n_frames)
        # src = src.transpose(0, 2, 1)
        batch_size, seq_len, freq = src.shape
        src = np.reshape(src, (num_wav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src


    def soft_time_freq_mask(target_src, remaining_src):
        #                  |target_src|
        # mask =  -------------------------------
        #         |target_src| + |rem_src| + eps
        mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
        return mask

    skip = 224 // 4
    to_process = audio.FRAMES * 4
    steps = spect.shape[1] - to_process
    half_seq_phase = SEQ_LEN // 2
    print('seq_len:', half_seq_phase)
    while True:
        if i >= steps:
            break
        print(i, steps, str(datetime.datetime.now() - start))
        s = spect[:, i:i+to_process]
        
        batch = spec_to_batch(s.reshape((1, audio.ROWS, to_process)))
        _t = sess.run(to_run, feed_dict={x_mixed: batch})
        y1 = batch_to_spec(_t[0][0], 1)[:, :, :to_process].reshape((audio.ROWS, to_process))
        y2 = batch_to_spec(_t[0][1], 1)[:, :, :to_process].reshape((audio.ROWS, to_process))

        # mask_src1 = soft_time_freq_mask(y1, y2)
        # mask_src2 = 1. - y1
        mask_src1 = y1
        mask_src2 = y2
        # y1 = (s.reshape((224, to_process)) * mask_src1)
        # y2 = (s.reshape((224, to_process)) * mask_src2)
        
        logits0_snips[:, i:i+to_process] = mask_src1
        logits1_snips[:, i:i+to_process] = mask_src2

        s2 = spect[:, i+half_seq_phase:i+to_process+half_seq_phase]
        batch = spec_to_batch(s2.reshape((1, audio.ROWS, to_process)))
        _t = sess.run(to_run, feed_dict={x_mixed: batch})
        y1 = batch_to_spec(_t[0][0], 1)[:, :, :to_process].reshape((audio.ROWS, to_process))
        y2 = batch_to_spec(_t[0][1], 1)[:, :, :to_process].reshape((audio.ROWS, to_process))

        for ib in range(y2.shape[1]):
            if abs(ib % -SEQ_LEN) < 2 or abs(ib % SEQ_LEN) < 2:
                continue
            # import pdb; pdb.set_trace()
            # print('[+] copying over border:', ib)
            # print('[+] equals:', np.all(logits1_snips[:, ib+half_seq_phase] == y2[:, ib]))
            # logits1_snips[:, ib+half_seq_phase] = y2[:, ib]

        # print(logits1_snips.min(), logits1_snips.max())
        
        print(i)
        if '-v' in sys.argv:
            plt.figure(1)
            plt.subplot(2, 3, 1)
            plt.imshow(s.reshape((audio.ROWS, to_process)), cmap=cmap, vmin=0.0, vmax=1.0)
            plt.subplot(2, 3, 2)
            plt.imshow(y1, cmap=cmap, vmin=0.0, vmax=1.0)
            plt.subplot(2, 3, 3)
            plt.imshow(y2, cmap=cmap, vmin=0.0, vmax=1.0)
            plt.subplot(2, 3, 4)
            plt.imshow(batch_to_spec(batch, 1).squeeze(0), cmap=cmap, vmin=0.0, vmax=1.0)
            plt.subplot(2, 3, 5)
            plt.imshow(mask_src1, cmap='hot', vmin=0.0, vmax=1.0)
            plt.subplot(2, 3, 6)
            plt.imshow(mask_src2, cmap='hot', vmin=0.0, vmax=1.0)
            plt.figure(2, figsize=(20, 2))
            plt.subplot(1, 1, 1)
            # plt.figure(figsize=(20, 2))
            plt.imshow(logits1_snips, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)
            plt.tight_layout()
            plt.draw()
            plt.show(block=False)  # , plt.draw(), plt.show()
            plt.pause(0.01)
        
        i += to_process

    pickle.dump(logits0_snips, open(l0_out_filename, 'wb'))
    pickle.dump(logits1_snips, open(l1_out_filename, 'wb'))

# for l, name in [(logits0_snips, '.l0'), (logits1_snips, '.l1')]:
for l, name in [(logits1_snips, '.l1')]:
    out_filename = filename + name + weights_name
    print('[+] doing:', name)
    print('  [+] masked')
    # vocs = np.concatenate(l, axis=1)  
    # l = (l - l.min()) / (l.max() - l.min())

    # l = (l - l.min()) / (l.max() - l.min())
    
    # l = l ** 4
    # gamma = 0.2
    # inv_gamma = 1.0 / gamma
    # l = l ** inv_gamma
    # l = (l - l.min()) / (l.max() - l.min())

    # if '-v' in sys.argv:
    #     gamma = 2.0
    #     while gamma > 0:
    #         print('gamma: ', gamma)
    #         inv_gamma = 1.0 / gamma
    #         nl = l ** inv_gamma
    #         nl = (nl - nl.min()) / (nl.max() - nl.min())
    #         nl = (nl + l) / 2.0
    #         plt.subplot(1, 1, 1)
    #         plt.imshow(nl, cmap='hot', aspect=20, vmin=0.0, vmax=1.0)
    #         plt.tight_layout()
    #         plt.draw()
    #         plt.show(block=False)  # , plt.draw(), plt.show()
    #         plt.pause(0.01)
    #         gamma *= 0.9
    # gamma = 0.1
    # inv_gamma = 1.0 / gamma
    # nl = l ** inv_gamma
    # nl = (nl - nl.min()) / (nl.max() - nl.min())
    # nl = (nl + l) / 2.0
    # l = nl
    def mix(a, b, factor):
        return a * (1.0 - factor) + b * factor
    
    
    def normalize(x):
        xmax = x.max()
        if xmax == 0:
            return x
        return x / xmax
    
    # maskmask = l < 0.6
    # l[maskmask] *= 0.1
    # l = gaussian_filter(l, sigma=3)
    # l = normalize(l)
    # l = (l - l.min()) / (l.max() - l.min())
    # l = np.clip(l, 0.9, 1.0)

    if '-v' in sys.argv:
        plt.subplot(1, 1, 1)
        plt.imshow(l, cmap='hot', aspect=20, vmin=0.0, vmax=1.0)
        # plt.pcolormesh(l, cmap='hot', vmin=0.0, vmax=1.0)
        plt.tight_layout()
        plt.draw()
        plt.show()

    plt.axis('off')
    out = l.copy()
    # for _ in range(20):
    #     print(_)
        # out = mix(gaussian_filter1d(out, 10, axis=1), l, 0.5)
    # out[out < 0.55] = 0
    # out = normalize(l + (gaussian_filter1d(out, 20, axis=1) - out))
    # out = normalize(gaussian_filter1d(out, 20, axis=1))
    # out = normalize(gaussian_filter1d(out, 5, axis=0))
    # nl = l.copy()
    # nl[nl < 0.35] = 0
    # out = mix(out, normalize(nl), 0.5)
    plt.imshow(out, cmap='hot', aspect=20, vmin=0.0, vmax=1.0)
    plt.savefig(name, dpi=300, orientation='landscape', bbox_inches='tight')

    l = out

    vocs = l
    # voc_reg = np.dot(np.transpose(mel_basis), vocs)
    voc_reg = vocs
    FACTOR = 3
    voc_reg[voc_reg < 0.35] *= 0.01
    # voc_reg = normalize(voc_reg)
    # voc_reg[voc_reg < 0.5] *= 0.5
    # voc_reg[voc_reg < 0.5] *= 0.5
    # voc_reg[voc_reg < 0.3] *= 0.5
    # voc_reg[voc_reg < 0.3] *= 0.3
    # voc_reg[voc_reg < 0.1] *= 0.1
    # voc_reg = normalize(mix(gaussian_filter1d(voc_reg, 10, axis=1), voc_reg, 0.5))
    # voc_reg = normalize(mix(gaussian_filter1d(voc_reg, 10, axis=1), voc_reg, 0.5))
    # voc_reg = normalize(mix(gaussian_filter1d(voc_reg, 10, axis=1), voc_reg, 0.5))
    # voc_reg[120:] = 0
    # voc_reg = normalize(gaussian_filter1d(voc_reg, 1, axis=0))
    # voc_reg = normalize(gaussian_filter(voc_reg, sigma=2))
    # voc_reg = normalize((voc_reg * FACTOR) ** FACTOR)
    # voc_reg[voc_reg < 0.05] = 0
    # masked = mag.T[:voc_reg.shape[1]].T * gaussian_filter(voc_reg, sigma=1)
    masked = mag.T[:voc_reg.shape[1]].T * normalize(voc_reg)

    # masked[masked < 0.2] *= masked[masked < 0.2]
    # masked[masked < 0.1] = 0.0
    S2 = masked * np.exp(1.j * phase.T[:voc_reg.shape[1]].T)
    y2 = librosa.istft(S2, audio.HOP_LENGTH)
    librosa.output.write_wav(out_filename + '.masked.vocl.wav', y2, audio.SAMPLE_RATE, norm=False)
    print('      ' + out_filename + '.masked.vocl.wav')
    # print('  [+] regular')
    # S2 = voc_reg * np.exp(1.j * phase.T[:voc_reg.shape[1]].T)
    # y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)
    # librosa.output.write_wav(out_filename + '.regular.vocl.wav', y2, sr, norm=False)
    # print('      ' + out_filename + '.regular.vocl.wav')


# vocs = np.concatenate(preds_snips, axis=1)
# voc_reg = np.dot(np.transpose(mel_basis), vocs)

# masked = mag.T[:voc_reg.shape[1]].T * gaussian_filter(voc_reg, sigma=1)
# S2 = masked * phase.T[:voc_reg.shape[1]].T
# y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)

# librosa.output.write_wav(filename + '.vocl.wav', y2, sr, norm=False)
