import numpy as np
import librosa
import audio
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import logging
import sys


from functools import partial
from rnn import RNNModel
from scipy.ndimage.filters import gaussian_filter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.WARNING)



filename = '/Users/amiramitai/Projects/nomix/02 Made In Heaven.mp3'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-06_085120'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-06_104506'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-06_171143'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-09_064212'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-09_214438'
# weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-10_171313'
weights = '/Users/amiramitai/Projects/nomix/rnn-2018-05-10_221942'


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

y, sr = librosa.load(filename, sr=audio.SAMPLE_RATE)
win_length = audio.FFT
if y.ndim > 1:
    y = y[0]
S = librosa.stft(y, audio.FFT, hop_length=audio.HOP_LENGTH, win_length=win_length)
mag, phase = librosa.magphase(S)

# mel spectrum
mel_basis = librosa.filters.mel(sr, audio.FFT, n_mels=audio.MELS)
mel = np.dot(mel_basis, mag)
print(mel.shape, mag.shape)

mel_graph = librosa.power_to_db(mel ** 2.0, ref=np.max)
mel_graph = (mel_graph.clip(-80, 0) + 80) / 80.0
print('mel_graph.shape:', mel_graph.shape)

# plt.imshow(feature, cmap='Greys_r')
# plt.show()
SEQ_LEN = 12
IM_HEIGHT = 224
IM_WIDTH = 224
IM_CHANNEL = 1
models_to_keep = 3
models_history = []
checkpoint_steps = 250
model_version = 1
K = IM_WIDTH * 64
num_classes = 2

with tf.name_scope('tower_0') as scope:
    pass

x_mixed = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='x_mixed')
y_src1 = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='y_src1')
y_src2 = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='y_src2')
global_step = tf.Variable(0, trainable=False)
net = RNNModel(x_mixed, y_src1, y_src2, hidden_size=1024, seq_len=SEQ_LEN, training=False)


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
    if name.startswith('RNN/'):
        name = name[4:]
    name = name.split(':')[0]
    if name in var_to_shape_map:
        print('[+] assigning:', v.name)
        # import bpdb; bpdb.set_trace()
        ops.append(tf.assign(v, reader.get_tensor(name)))
    else:
        print('[+] skipping:', v.name)
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
# import pdb; pdb.set_trace()

preds_snips = []
logits0_snips = []
logits1_snips = []
to_run = [net.logits]

cmap = 'hot'
# cmap = 'Greys_r'
# for i, s in enumerate(snippets):
logits0_snips = np.zeros(mel_graph.shape)
logits1_snips = np.zeros(mel_graph.shape)
steps = mel_graph.shape[1] - 224
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
while True:
    if i >= steps:
        break
    s = mel_graph[:, i:i+224]
    # import pdb; pdb.set_trace()
    # print(s.min(), s.max())
    # batch = []
    print(i, steps, str(datetime.datetime.now() - start))
    batch = spec_to_batch(s.reshape((1, 224, 224)))

    # print(batch.shape)
    _t = sess.run(to_run, feed_dict={x_mixed: batch})
    y1 = batch_to_spec(_t[0][0], 1)[:, :, :224].reshape((224, 224))
    y2 = batch_to_spec(_t[0][1], 1)[:, :, :224].reshape((224, 224))

    # mask_src1 = soft_time_freq_mask(y1, y2)
    # mask_src2 = 1. - y1
    mask_src1 = y1
    mask_src2 = y2
    y1 = (s.reshape((224, 224)) * mask_src1)
    y2 = (s.reshape((224, 224)) * mask_src2)
    
    logits0_snips[:, i+skip:i+2*skip] = y1[:, skip:2*skip]
    logits1_snips[:, i+skip:i+2*skip] = y2[:, skip:2*skip]

    # print(logits1_snips.min(), logits1_snips.max())
    
    print(i)
    if '-v' in sys.argv:
        plt.figure(1)
        plt.subplot(2, 3, 1)
        plt.imshow(s.reshape((224, 224)), cmap=cmap, vmin=0.0, vmax=1.0)
        plt.subplot(2, 3, 2)
        plt.imshow(y1, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.subplot(2, 3, 3)
        plt.imshow(y2, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.subplot(2, 3, 4)
        plt.imshow(batch_to_spec(batch, 1).squeeze(0), cmap=cmap, vmin=0.0, vmax=1.0)
        plt.subplot(2, 3, 5)
        plt.imshow(mask_src1, cmap='Greys_r', vmin=0.0, vmax=1.0)
        plt.subplot(2, 3, 6)
        plt.imshow(mask_src2, cmap='Greys_r', vmin=0.0, vmax=1.0)
        plt.figure(2, figsize=(20, 2))
        plt.subplot(1, 1, 1)
        # plt.figure(figsize=(20, 2))
        plt.imshow(logits1_snips, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)
        plt.tight_layout()
        plt.draw()
        plt.show(block=False)  # , plt.draw(), plt.show()
        plt.pause(0.01)
    
    i += skip

# import pdb; pdb.set_trace()

# plt.show()

 # Time-frequency masking


weights_name = '.' + weights.split('_')[-1]
for l, name in [(logits0_snips, '.l0'), (logits1_snips, '.l1')]:
    out_filename = filename + name + weights_name
    print('[+] doing:', name)
    print('  [+] masked')
    # vocs = np.concatenate(l, axis=1)  
    vocs = l
    voc_reg = np.dot(np.transpose(mel_basis), vocs)
    # masked = mag.T[:voc_reg.shape[1]].T * gaussian_filter(voc_reg, sigma=1)
    masked = mag.T[:voc_reg.shape[1]].T * voc_reg
    S2 = masked * np.exp(1.j * phase.T[:voc_reg.shape[1]].T)
    y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)
    librosa.output.write_wav(out_filename + '.masked.vocl.wav', y2, sr, norm=False)
    print('      ' + out_filename + '.masked.vocl.wav')
    print('  [+] regular')
    S2 = voc_reg * np.exp(1.j * phase.T[:voc_reg.shape[1]].T)
    y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)
    librosa.output.write_wav(out_filename + '.regular.vocl.wav', y2, sr, norm=False)
    print('      ' + out_filename + '.regular.vocl.wav')


# vocs = np.concatenate(preds_snips, axis=1)
# voc_reg = np.dot(np.transpose(mel_basis), vocs)

# masked = mag.T[:voc_reg.shape[1]].T * gaussian_filter(voc_reg, sigma=1)
# S2 = masked * phase.T[:voc_reg.shape[1]].T
# y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)

# librosa.output.write_wav(filename + '.vocl.wav', y2, sr, norm=False)
