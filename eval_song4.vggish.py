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

    default_device = '/gpu:0'
    # default_device = '/cpu:0'

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
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
        # please do not use the totality of the GPU memory
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        config.gpu_options.allow_growth = True
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
                
                tvars = tf.trainable_variables() 
                for v in tvars:
                    print(v)
                    if 'weights' in v.name:
                        heavy_sum.append(tf.summary.histogram(v.name, v))
                        # if 'conv1_1' in v.name:
                        #     light_sum.append(tf.summary.histogram(v.name, v))
                for v in tvars:
                    if 'bias' in v.name:
                        heavy_sum.append(tf.summary.histogram(v.name, v))
                        # if 'conv1_1' in v.name:
                        #     light_sum.append(tf.summary.histogram(v.name, v))
                
                light_sum.append(tf.summary.histogram("predictions", predictions))
                light_sum.extend(model.summaries)
                heavy_sum.extend(model.heavy_summaries)

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

                tvars = tf.trainable_variables() 
                L2 = [tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]
                lossL2 = tf.add_n(L2) * 0.01

                cost = tf.reduce_mean(cross_entropy, name='cost') + lossL2
                # cost = tf.reduce_mean(cross_entropy, name='cost')

                light_sum.append(tf.summary.scalar("cost", cost))

            def my_capper(t):
                print(t)
                # return t
                if t is None:
                    return None
                
                # return tf.clip_by_value(t, 0., 5.)
                return tf.clip_by_value(t, -5., 5.)
                # return t

            with tf.name_scope("0_train"):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):         

                    vars_to_train = []
                    for v in tf.trainable_variables():
                        # if v is not None and 'conv1_1' in v.name:
                        #     continue
                        
                        vars_to_train.append(v)

                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    grads_and_vars = optimizer.compute_gradients(cost, var_list=vars_to_train)
                    grads_and_vars = [(my_capper(gv[0]), gv[1]) for gv in grads_and_vars]
                    optimizer = optimizer.apply_gradients(grads_and_vars)
                    
                    correct_predictions = tf.equal(tf.argmax(predictions, 1),
                                                   tf.argmax(_labels, 1),
                                                   name='correct_predictions')
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                              name='accuracy')

                    grad_norm = tf.norm(grads_and_vars[0][0])
                    light_sum.append(tf.summary.scalar("accuracy", accuracy))
                    light_sum.append(tf.summary.scalar("gradient", grad_norm))
                    # tf.summary.scalar("learning_rate", learning_rate)

            light_summary = tf.summary.merge(light_sum)
            heavy_summary = tf.summary.merge(light_sum + heavy_sum)
            sess.run(tf.global_variables_initializer())
            
            # test_batches = get_batches('TEST', batch_size=batch_size, repeat=True)

            log_string = 'logs/{}-vggish/{}-lr-{:.8f}'.format(model_version,
                                                              time_str,
                                                              learning_rate)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            try:
                print('[+] loading startup.json')
                startup = json.load(open('startup.vggish.json', 'r'))
                print('[+] loading path:', startup['path'])
                state = json.load(open(startup['path'], 'r'))
                print('[+] loading checkpoint:', state['checkpoint_path'])
                last_checkpoint = os.path.dirname(state['checkpoint_path'])
                saver.restore(sess, tf.train.latest_checkpoint(last_checkpoint))
                # iteration = state['iteration']
                # best_acc = state['best_acc']
                # best_loss = state['best_loss']
                # best_acc = state['best_acc']
                # if 'val_acc' in state:
                #     best_acc = state['best_acc']

                best_loss = state['best_loss']
                if 'train_loss' in state:
                    best_loss = state['best_loss']

                checkpoint_path = state['checkpoint_path']
                # log_string = 'next-' + state['log_string']
            except:
                print('[!] no models to checkpoint from..')
                raise
            writer = tf.summary.FileWriter(log_string)
            # import numpy as np
            # vcol = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # for v in vcol:
            #     res = sess.run([v])
            #     np.save(v.name.replace('/', '_').replace(':', '_')+'.npy', res)

            seq = iaa.Sequential(
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.Noop(),
                    # iaa.GaussianBlur(sigma=(0.5, 2.0)),
                    # iaa.Add((-50.0, 50.0), per_channel=False),
                    # iaa.AdditiveGaussianNoise(loc=0,
                    #                           scale=(0.07*255, 0.07*255),
                    #                           per_channel=False),
                    # iaa.Dropout(p=0.07, per_channel=False),
                    # iaa.CoarseDropout(p=(0.05, 0.15),
                    #                   size_percent=(0.1, 0.9),
                    #                   per_channel=False),
                    # iaa.SaltAndPepper(p=(0.05, 0.15), per_channel=False),
                    # iaa.Salt(p=(0.05, 0.15), per_channel=False),
                    # iaa.Pepper(p=(0.05, 0.15), per_channel=False),
                    # iaa.ContrastNormalization(alpha=(iap.Uniform(0.02, 0.03),
                    #                           iap.Uniform(1.7, 2.1))),
                    # iaa.ElasticTransformation(alpha=(0.5, 2.0)),
                ]),
            )

            
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
                # for i in range(batch_size):
                #     f, l = tip.get()
                #     features.append(f.reshape((224, 224, 1)))
                #     labels.append(l)
                
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
            # train = cc.random_iterator(batch_size, test=False)
            # start_learning_rate = (1e-05 + 1e-06) / 2.0
            # start_learning_rate = 1e-02
            # end_learning_rate = (1e-07 + 1e-06) / 2.0
            #end_learning_rate = start_learning_rate
            # accuracy_print_steps = 20
            light_summary_steps = 5
            heavy_summary_steps = light_summary_steps * 10
            checkpoint_steps = 100
            steps_to_decay = 30000.0
            best_interval = 0.00001
            iteration = 0
            best_loss = 0.6
            batch_num = 0
            best_acc = 0
            grad = 5
            models_history = []
            models_to_keep = 5

            def mix(a, b, mix_factor):
                return a * (1.0 - mix_factor) + b * mix_factor
            # check_op = tf.add_check_numerics_ops()
            cur_time = 0
            while True:
                if iters and iteration >= iters:
                    return
                
                seq.reseed(seed())
                np.random.seed(seed())


                # if grad < 0.3:
                #     to_noise = 12
                #     features, labels = get_data_from_tip(tip, batch_size - to_noise)
                #     noise_f = np.random.rand(to_noise, 224, 224, 1)
                #     noise_l = np.array([[0, 1]] * (to_noise // 2) + [[1, 0]] * (to_noise // 2))
                #     print(features.shape, noise_f.shape)
                #     features = np.concatenate([features, noise_f])
                #     labels = np.concatenate([labels, noise_l])
                # else:
                features, labels = get_data_from_tip(tip, batch_size)
                iteration += 1
                
                p, = sess.run(
                    [predictions],
                    feed_dict={
                        _images: seq.augment_images(features * 255) / 255,
                        # _images: features,
                        # _labels: labels,
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
                    print('\t{}. a:{} p:{} - gt:{}'.format(a, format_secs(cur_time), pred, gt))
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
    while True:
        # for lr in [1e-04, 1e-05, 1e-06]:
        for lr in [1e-06]:
            try:
                train(tip, iters=None, learning_rate=lr, batch_norm=True)
            except KeyboardInterrupt:
                should_quit = False
                while True:
                    ans = input("should quit? [y/n]:")
                    if ans.lower() in ['y', 'n']:
                        should_quit = ans.lower() == 'y'
                        break
                if should_quit:
                    print('bye!')
                    break
            except:
                traceback.print_exc()
                tb = traceback.format_exc()
                send_mail(subject="Training Failed!", body=tb)
                raise
        # train(tip, iters=3000, learning_rate=lr, batch_norm=True)
        # train(tip, iters=3000, learning_rate=lr, batch_norm=True)

    # train(tip, iters=700, learning_rate=0.9*3, batch_norm=True)
    # train(tip, iters=700, learning_rate=0.9*3*3, batch_norm=True)
    
    # train(tip, iters=7000, learning_rate=1e-03, batch_norm=True)
    # train(tip, iters=7000, learning_rate=1e-04, batch_norm=True)
    # train(tip, iters=7000, learning_rate=1e-05, batch_norm=True)
    # train(tip, iters=4000, learning_rate=0.0001)
    # train(tip, iters=4000, learning_rate=0.00001)
    # train(tip, iters=4000, learning_rate=0.000001)
    # cc = cache_file.CacheCollection({'filename': 'T:\\cache\\AudioToImage', 'seek_policy': 'ONE_SHOT', 'max_size': 2147483648, 'max_split': 50})
    # print(cc.mean())