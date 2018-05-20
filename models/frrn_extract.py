import numpy as np
import librosa
import audio
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time


from functools import partial
from frrn import _arch_type_a, FRRN
from scipy.ndimage.filters import gaussian_filter

filename = '/Users/amiramitai/Projects/nomix/02 Made In Heaven.mp3'
# weights = '/Users/amiramitai/Projects/nomix/frrn-2018-04-20_171751'
# weights = '/Users/amiramitai/Projects/nomix/frrn-2018-04-20_210422'
# weights = '/Users/amiramitai/Projects/nomix/frrn-2018-04-21_080201'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-21_123623'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-21_131920'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-21_151957'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-21_180937'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-21_203155'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-21_234400'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-22_083346'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-22_190104'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-23_080143'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-23_165247'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-23_225340'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-24_071736'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-25_184136'
# weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-04-28_194227'
weights = '/Users/amiramitai/Projects/nomix/frrn2-2018-05-04_123043'


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
mel_graph = mel_graph.clip(-80, 0) + 80
print('mel_graph.shape:', mel_graph.shape)

# plt.imshow(feature, cmap='Greys_r')
# plt.show()

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

im = tf.placeholder(tf.float32, [None, IM_HEIGHT, IM_WIDTH, IM_CHANNEL])
gt = tf.placeholder(tf.int32, [None, IM_HEIGHT, IM_WIDTH])
global_step = tf.Variable(0, trainable=False)
net = FRRN(K, im, gt, partial(_arch_type_a, num_classes), scope)


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
# latest = tf.train.latest_checkpoint(weights)
s.restore(sess, latest)

# snippets = split_spectrogram_into_snippets(mel_graph, 224)
# import pdb; pdb.set_trace()

preds_snips = []
logits0_snips = []
logits1_snips = []
to_run = [net.logits, net.preds]

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
skip = 224 // 4
while True:
    if i >= steps:
        break
    s = mel_graph[:, i:i+224]
    # batch = []
    print(i, steps, str(datetime.datetime.now() - start))
    _t, preds = sess.run(to_run, feed_dict={im: s.reshape((1, IM_HEIGHT, IM_WIDTH, 1))})

    # logits0_snips[:, i:i+224] += _t[0, :, :, 0] / (224 / skip)
    # logits1_snips[:, i:i+224] += _t[0, :, :, 1] / (224 / skip)
    logits0_snips[:, i+skip:i+2*skip] += _t[0, :, skip:2*skip, 0]
    logits1_snips[:, i+skip:i+2*skip] += _t[0, :, skip:2*skip, 1]
    # if i % 5 == 0:
    #     plt.figure(1)
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(s.reshape((224, 224)), cmap=cmap)
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(_t[:, :, :, 0].reshape((224, 224)), cmap=cmap)
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(_t[:, :, :, 1].reshape((224, 224)), cmap=cmap)
    #     plt.figure(2, figsize=(20, 2))
    #     plt.subplot(1, 1, 1)
    #     # plt.figure(figsize=(20, 2))
    #     plt.imshow(logits1_snips, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)
    #     plt.tight_layout()
    #     plt.draw()
    #     plt.show(block=False)  # , plt.draw(), plt.show()
    #     plt.pause(0.01)
    
    i += skip

# import pdb; pdb.set_trace()

# plt.show()

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
    S2 = masked * phase.T[:voc_reg.shape[1]].T
    y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)
    librosa.output.write_wav(out_filename + '.masked.vocl.wav', y2, sr, norm=False)
    print('  [+] regular')
    S2 = voc_reg * phase.T[:voc_reg.shape[1]].T
    y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)
    librosa.output.write_wav(out_filename + '.regular.vocl.wav', y2, sr, norm=False)


# vocs = np.concatenate(preds_snips, axis=1)
# voc_reg = np.dot(np.transpose(mel_basis), vocs)

# masked = mag.T[:voc_reg.shape[1]].T * gaussian_filter(voc_reg, sigma=1)
# S2 = masked * phase.T[:voc_reg.shape[1]].T
# y2 = librosa.istft(S2, hop_length=audio.HOP_LENGTH, win_length=win_length)

# librosa.output.write_wav(filename + '.vocl.wav', y2, sr, norm=False)
