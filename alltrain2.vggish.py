import os
import time
import math

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

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

from imgaug import augmenters as iaa
from imgaug import parameters as iap

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


def seed():
    return int(time.time() * 10000000) % (2**32-1)

def get_data_from_tip(tip, batch_size):
    features = []
    labels = []
    descriptions = []
    for i in range(batch_size):
        data = tip.get()
        d, f, l = data
        features.append(f.reshape((224, 224, 1)))
        labels.append(l)
        descriptions.append(d)

    return descriptions, np.array(features), np.array(labels)

def mix(a, b, mix_factor):
    return a * (1.0 - mix_factor) + b * mix_factor


def train(tip, iters=None, learning_rate=0.001, batch_norm=False):
    import tensorflow as tf

    random.seed(datetime.datetime.now())
    tf.set_random_seed(seed())

    default_device = '/gpu:0'
    # default_device = '/cpu:0'

    # hyperparams
    batch_size = 94
    training = True
    batch_norms_training = False
    # steps
    light_summary_steps = 25
    heavy_summary_steps = 250
    checkpoint_steps = 500
    # stats / logging
    model_version = 1
    time_str = datetime.datetime.now().strftime('%m-%d--%H%M%S')
    best_loss = 0.6
    batch_num = 0
    best_acc = 0
    models_to_keep = 3
    # glob vars
    heavy_sum = []
    light_sum = []
    models_history = []
    train_only_dense = False
    dense_size = 64
    dropout = True
    learning_rate = 1e-6
    augmentation = True    

    with tf.device(default_device):
        # config = tf.ConfigProto()
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
        # please do not use the totality of the GPU memory
        config.gpu_options.per_process_gpu_memory_fraction = 0.98
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            with tf.name_scope("inputs"):
                # _images = tf.placeholder(tf.float32, [None, 224, 224, 1])
                _images = tf.placeholder(tf.float32, [batch_size, 224, 224, 1])
                _is_training = tf.placeholder(tf.bool, name='is_training')
            model = vggish.Vggish(_images,
                                  classes=2,
                                  trainable=training,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  only_dense=train_only_dense,
                                  dense_size=dense_size,
                                  bn_trainable=batch_norms_training)

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
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits,
                    labels=_labels,
                    name='cross_entropy'
                )

                tvars = tf.trainable_variables() 
                L2 = [tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name]
                lossL2 = tf.add_n(L2) * 0.01

                cost = tf.reduce_mean(cross_entropy, name='cost') + lossL2
                light_sum.append(tf.summary.scalar("cost", cost))

            def my_capper(t):
                print(t)
                # return t
                if t is None:
                    return None
                
                return tf.clip_by_value(t, -5., 5.)
            
            log_string = 'logs/{}-vggish/{}-lr-{:.8f}'.format(model_version,
                                                              time_str,
                                                              learning_rate)
            with tf.name_scope("0_train"):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):         
                    vars_to_train = model.get_vars_to_train()

                    global_step = tf.Variable(0, name='global_step')
                    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.96, staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                    grads_and_vars = optimizer.compute_gradients(cost, var_list=vars_to_train)
                    grads_and_vars = [(my_capper(gv[0]), gv[1]) for gv in grads_and_vars]
                    optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    
                    correct_predictions = tf.equal(tf.argmax(predictions, 1),
                                                   tf.argmax(_labels, 1),
                                                   name='correct_predictions')
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                              name='accuracy')

                    grad_norm = tf.norm(grads_and_vars[0][0])
                    light_sum.append(tf.summary.scalar("accuracy", accuracy))
                    light_sum.append(tf.summary.scalar("gradient", grad_norm))

            light_summary = tf.summary.merge(light_sum)
            heavy_summary = tf.summary.merge(light_sum + heavy_sum)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            iteration = 0

            try:
                print('[+] loading startup.json')
                startup = json.load(open('startup.vggish.json', 'r'))
                print('[+] loading path:', startup['path'])
                state = json.load(open(startup['path'], 'r'))
                print('[+] loading checkpoint:', state['checkpoint_path'])
                last_checkpoint = os.path.dirname(state['checkpoint_path'])
                
                weights_to_load = vggish.conv_vars + vggish.fc_vars
                if train_only_dense:
                    weights_to_load = vggish.conv_vars
                model.load_weights(weights_path, vars_names=weights_to_load)
                
                iteration = state['iteration']
                best_loss = state['best_loss']
                if 'train_loss' in state:
                    best_loss = state['best_loss']

                checkpoint_path = state['checkpoint_path']
            except:
                print('[!] no models to checkpoint from..')
                raise
            writer = tf.summary.FileWriter(log_string)


            augmenters = [
                # blur images with a sigma between 0 and 3.0
                iaa.Noop(),
                iaa.GaussianBlur(sigma=(0.5, 2.0)),
                iaa.Add((-50.0, 50.0), per_channel=False),
                iaa.AdditiveGaussianNoise(loc=0,
                                            scale=(0.07*255, 0.07*255),
                                            per_channel=False),
                iaa.Dropout(p=0.07, per_channel=False),
                iaa.CoarseDropout(p=(0.05, 0.15),
                                    size_percent=(0.1, 0.9),
                                    per_channel=False),
                iaa.SaltAndPepper(p=(0.05, 0.15), per_channel=False),
                iaa.Salt(p=(0.05, 0.15), per_channel=False),
                iaa.Pepper(p=(0.05, 0.15), per_channel=False),
                iaa.ContrastNormalization(alpha=(iap.Uniform(0.02, 0.03),
                                            iap.Uniform(1.7, 2.1))),
                iaa.ElasticTransformation(alpha=(0.5, 2.0)),
            ]

            if not augmentation:
                augmenters = [
                    iaa.Noop(),
                ]
            
            seq = iaa.Sequential(
                iaa.OneOf(augmenters),
            )

            while True:
                if iters and iteration >= iters:
                    return
                
                seq.reseed(seed())
                np.random.seed(seed())

                descriptions, features, labels = get_data_from_tip(tip, batch_size)
                iteration += 1
                # import pdb; pdb.set_trace()
                mean = 0.172840994091
                std = 0.206961060284
                merged_summaries = light_summary
                if iteration % heavy_summary_steps == 0:
                    merged_summaries = heavy_summary

                try:
                    with np.errstate(all='raise'):
                        for i in range(5):
                            newfeatures = seq.augment_images(features * 255) / 255
                            if not np.isnan(newfeatures).any():
                                break
                            print('[!] has nan in newfeatures, retrying', i)
                        if np.isnan(newfeatures).any():
                            print('[!] could not get rid of nan.. skipping this batch')
                            iteration -= 1
                            continue
                        features = newfeatures
                except Exception:
                    print("[!] Warning detected augmenting, skipping..")
                    tb = traceback.format_exc()
                    open("numpy_warns.log", 'ab').write(str(descriptions).encode('utf-8'))
                    open("numpy_warns.log", 'ab').write(str(tb).encode('utf-8'))
                    open("numpy_warns.log", 'a').write('------------------------------------')
                    # import pdb; pdb.set_trace()
                    continue

                feed_dict = {
                    _images: features,
                    _labels: labels,
                    _is_training: training
                }

                train_loss, val_acc, _, p, summary, grad, _corr_pred = sess.run(
                    [cost,
                     accuracy,
                     optimizer,
                     logits,
                     merged_summaries,
                     grad_norm,
                     correct_predictions],
                    feed_dict=feed_dict
                )

                print('[+] iteration {}'.format(iteration))
                if iteration % light_summary_steps == 0:
                    if os.path.isfile('nomix_pdb'):
                        import pdb
                        pdb.set_trace()

                    print('[+] writing summary')
                    writer.add_summary(summary, iteration)

                    print('\tIteration {} Accuracy: {} Loss: {}/{}'.format(iteration,
                                                                           val_acc,
                                                                           train_loss,
                                                                           best_loss))
                # if train_loss < best_loss:
                if iteration % checkpoint_steps == 0:
                    # print('\t\tNew Best Loss!')
                    best_loss = train_loss
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    
                    checkpoint_dir = os.path.join('D:\\checkpoint', timestamp)
                    checkpoint_path = os.path.join('D:\\checkpoint', timestamp, 'model.ckpt')
                    print('\t\tSaving model to:' + checkpoint_path)
                    saver.save(sess, checkpoint_path, global_step=batch_num)
                    state = {
                        'iteration': iteration,
                        'best_acc': float(best_acc),
                        'best_loss': float(best_loss),
                        'val_acc': float(val_acc),
                        'train_loss': float(train_loss),
                        'checkpoint_path': checkpoint_path,
                        'log_string': log_string,
                    }
                    # state_path = os.path.join('save', timestamp, 'state.json')
                    state_path = os.path.join('D:\\checkpoint', timestamp, 'state.json')
                    open(state_path, 'w').write(json.dumps(state))
                    startup = {
                        'path': state_path,
                    }
                    open('startup.vggish.json', 'w').write(json.dumps(startup))
                    models_history.append(checkpoint_dir)
                    while len(models_history) > models_to_keep:
                        try:
                            path_to_del = models_history.pop(0)
                            print('[+] deleting model', path_to_del)
                            shutil.rmtree(path_to_del)
                        except:
                            print('[+] failed to delete')
                            traceback.print_exc()
                

if __name__ == '__main__':
    from pipeline import Pipeline
    import yaml
    import matplotlib.pyplot as plt
    with open('configs/win_client.2.yaml') as f:
        config = yaml.safe_load(f)
    amp = Pipeline(config['pipeline'])
    amp.run()
    amp.keepalive(block=False)
    tip = amp._get_tip_queue()
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