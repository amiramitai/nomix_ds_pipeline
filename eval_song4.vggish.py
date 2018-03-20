import os
import time
import math

import numpy as np
import time
import random
import pickle
import datetime
import vggish
import cache_file
import json
import sys
import traceback
from utils import send_mail
import shutil
import audio


# import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

from imgaug import augmenters as iaa
from imgaug import parameters as iap

def get_imseq():
    import librosa
    import audio
    y, sr = librosa.load(sys.argv[1],
                         sr=audio.SAMPLE_RATE,
                         mono=False)

    if y.ndim > 1:
        y = y[0]

    mel = librosa.feature.melspectrogram(y=y,
                                         sr=audio.SAMPLE_RATE,
                                         n_mels=audio.MELS,
                                         n_fft=audio.FFT,
                                         power=audio.POWER,
                                         hop_length=audio.HOP_LENGTH)
    
    image = librosa.power_to_db(mel, ref=np.max)
    image = (image.clip(-80, 0) + 80) / 80
    imseq = image.T.flatten()[:image.size - image.size % (audio.MELS*audio.MELS)].reshape((-1, audio.MELS,audio.MELS))
    return imseq


def get_lines_with_file(f, nlines, repeat):
    ret = []
    for i in range(nlines):
        path = f.readline().strip()
        if not path:
            if repeat:
                f.seek(0)
                path = f.readline().strip()
            else:
                return ret
        ret.append(path)
    return ret

def seed():
    return int(time.time() * 10000000) % (2**32-1)

def train(tip, iters=None, learning_rate=0.001, batch_norm=False):
    import tensorflow as tf

    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model.signature_def_utils import predict_signature_def

    from tensorflow.python.saved_model.tag_constants import SERVING
    from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
    from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
    from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS

    random.seed(datetime.datetime.now())



    tf.set_random_seed(seed())

    # default_device = '/gpu:0'
    default_device = '/cpu:0'

    num_hidden_neurons = 256

    vgg_mean = [103.939, 116.779, 123.68]
    # model_version = int(time.time())
    model_version = 1
    model_path = 'models/model-{}/'.format(model_version)
    accuracy_print_steps = 100
    batch_size = 22
    # learning_rate = 0.001
    num_hidden_layers = 2
    hidden_layer_size = 512
    keep_prob = 0.6
    time_str = datetime.datetime.now().strftime('%m-%d--%H%M%S')
    training = False
    heavy_sum = []
    light_sum = []
    with tf.device(default_device):
        # config = tf.ConfigProto()
        # config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
        # please do not use the totality of the GPU memory
        # config.gpu_options.per_process_gpu_memory_fraction = 0.90
        # config.gpu_options.allow_growth = True
        config = tf.ConfigProto(device_count = {'GPU': 0})
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            with tf.name_scope("inputs"):
                # _images = tf.placeholder(tf.float32, shape=(None, 4096), name='images')
                _images = tf.placeholder(tf.float32, [None, 224, 224, 1])
                _is_training = tf.placeholder(tf.bool, name='is_training')
                _keep_prob = tf.placeholder(tf.float32, name='keep_probability')
            # imgs = tf.placeholder(tf.float32, [None, 224, 224, 1])
            # model = vggish.Vggish(_images,
            #                       classes=2,
            #                       mean=[0.343388929118],
            #                       trainable=training,
            #                       batch_norm=batch_norm)
            model = vggish.Vggish(_images,
                                  classes=2,
                                  mean=[0.343388929118],
                                  trainable=training,
                                  batch_norm=batch_norm)

            with tf.name_scope("targets"):
                _labels = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

            with tf.name_scope("outputs"):
                logits = model.fc3l
                # predictions = tf.nn.softmax(logits, name='predictions')
                predictions = tf.nn.sigmoid(logits, name='predictions')
                

            with tf.name_scope("0_cost"):
                # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                #   logits=logits,
                #   labels=_labels,
                #   name='cross_entropy'
                # )
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits,
                    labels=_labels,
                    name='cross_entropy'
                )


            sess.run(tf.global_variables_initializer())
            
            # test_batches = get_batches('TEST', batch_size=batch_size, repeat=True)

            log_string = 'logs/{}-vggish/{}-lr-{:.8f}'.format(model_version,
                                                              time_str,
                                                              learning_rate)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            last_checkpoint = "D:\\checkpoint\\2018-03-20_232627"
            saver.restore(sess, tf.train.latest_checkpoint(last_checkpoint))
            

            
            import matplotlib.pyplot as plt

            def get_next_imseq():
                imseq = get_imseq()
                fig = plt.figure()
                for im in imseq:
                    # a = fig.add_subplot(1, 1, 1)
                    # plt.imshow(im.reshape((audio.MELS, audio.MELS)), cmap='hot')
                    # plt.show()
                    yield im.T.reshape((audio.MELS, audio.MELS, 1))

            next_imseq = get_next_imseq()
            
            def get_data_from_tip(tip, batch_size):
                features = []
                labels = []
                
                for im in next_imseq:
                    features.append(im)
                    labels.append([1, 0])
                    if len(features) == batch_size:
                        break
                
                return np.array(features), np.array(labels)
                    

            def format_secs(secs):
                mins = int(secs / 60.0)
                secs -= mins * 60
                return '{}:{:02d}'.format(mins, int(secs))
            
            time_fract = (audio.MELS * audio.HOP_LENGTH) / audio.SAMPLE_RATE
            cur_time = 0
            for iteration in range(1000):
                features, labels = get_data_from_tip(tip, batch_size)

                if len(features) == 0:
                    return
                
                p, = sess.run(
                    [predictions],
                    feed_dict={
                        _images: features,
                        _is_training: training
                    })

                to_text_label = {
                    0: 'inst',
                    1: 'vocl'
                }

                print(iteration)
                for i, (a, l) in enumerate(zip(p, labels)):
                    pred = to_text_label[np.argmax(a)]
                    gt = to_text_label[np.argmax(l)]
                    print('\t{}. p:{} - gt:{} a:{} '.format(format_secs(cur_time), pred, gt, a))
                    cur_time += time_fract

                # writer.add_summary(summary, iteration)
                # global_step += 1
                # print('Iteration {} Accuracy: {} Loss: {}/{}'.format(iteration,
                #                                                      val_acc,
                #                                                      train_loss,
                #                                                      best_loss))

                

if __name__ == '__main__':
    from pipeline import Pipeline
    import yaml
    import matplotlib.pyplot as plt
    with open('configs/win_client.2.yaml') as f:
        config = yaml.safe_load(f)
    # amp = Pipeline(config['pipeline'])
    # amp.run()
    # amp.keepalive(block=False)
    # tip = amp._get_tip_queue()
    tip = None
    # while True:
    #     fig = plt.figure(figsize=(16,4))
    #     a=fig.add_subplot(1,1,1)
    #     a = tip.get()
    #     print("[+] Y:",a[1])
    #     plt.imshow(a[0], cmap='hot')
    #     plt.show()
    # train(tip, iters=3500, learning_rate=0.1)
    # train(tip, iters=4000, learning_rate=0.01)

    # train(tip, iters=7000, learning_rate=1e-01, batch_norm=True)
    # for lr in [1e-05, 1e-06, 1e-07, 1e-08]:
    train(tip, iters=None, learning_rate=0, batch_norm=True)