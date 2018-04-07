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
    steps_to_decay = 30000.0
    # steps
    light_summary_steps = 25
    heavy_summary_steps = 250
    accuracy_print_steps = 100
    checkpoint_steps = 500
    # stats / logging
    model_version = 1
    model_path = 'models/model-{}/'.format(model_version)
    time_str = datetime.datetime.now().strftime('%m-%d--%H%M%S')
    best_interval = 0.00001
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
                                  mean=[0.343388929118],
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
                
                return tf.clip_by_value(t, -5., 5.)
            
            log_string = 'logs/{}-vggish/{}-lr-{:.8f}'.format(model_version,
                                                              time_str,
                                                              learning_rate)
            load_batch_norms = False
            with tf.name_scope("0_train"):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):         
                    to_skip = []
                    vars_to_train = []
                    for v in tf.trainable_variables():
                        if v is None:
                            continue

                        if load_batch_norms and 'BatchNorm' in v.name:
                            print('[+] skipping:', v.name)
                            continue
                        
                        if 'BatchNorm' in v.name:
                            print('[+] skipping:', v.name)
                            continue
                        # if 'fc' not in v.name:
                        #     print('[+] skipping:', v.name)
                        #     continue

                        # print('[+] training:', v.name)
                        vars_to_train.append(v)

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

            # check_op = tf.add_check_numerics_ops()

            sess.run(tf.global_variables_initializer())
            
            # if load_batch_norms:
            #     batch_norms = []
            #     for v in tf.global_variables():
            #         if 'BatchNorm' not in v.name:
            #             continue
            #         batch_norms.append(v)

            #     # load batch norm
            #     saver = tf.train.Saver(batch_norms, max_to_keep=None)
            #     print('[+] loading batch norms layers')
            #     batch_norm_path = 'D:\\checkpoint\\2018-03-09_212452'
            #     saver.restore(sess, tf.train.latest_checkpoint(batch_norm_path))
            names_to_load = [
                'BatchNorm/beta:0',
                'conv1_1/weights',
                'conv1_1/biases',
                'BatchNorm_1/beta',
                'conv2_1/weights',
                'conv2_1/biases',
                'BatchNorm_2/beta',
                'conv3_1/weights',
                'conv3_1/biases',
                'BatchNorm_3/beta',
                'conv3_2/weights',
                'conv3_2/biases',
                'BatchNorm_4/beta',
                'conv4_1/weights',
                'conv4_1/biases',
                'BatchNorm_5/beta',
                'conv4_2/weights',
                'conv4_2/biases',
                'BatchNorm_6/beta'
            ]

            def should_load(name):
                for tl in names_to_load:
                    if tl in name:
                        return True

                return False

            if train_only_dense:
                to_load = []
                for v in tf.global_variables():
                    if not should_load(v.name):
                        continue
                    to_load.append(v)
                # load batch norm
                saver = tf.train.Saver(to_load, max_to_keep=None)
                print('[+] loading batch norms layers')
                batch_norm_path = 'D:\\checkpoint\\2018-03-23_193348'
                saver.restore(sess, tf.train.latest_checkpoint(batch_norm_path))

            vars_to_save = []
            for v in tf.global_variables():
                # if 'global_step' in v.name:
                #     continue
                vars_to_save.append(v)
            saver = tf.train.Saver(vars_to_save, max_to_keep=None)
            iteration = 0

            try:
                print('[+] loading startup.json')
                startup = json.load(open('startup.vggish.json', 'r'))
                print('[+] loading path:', startup['path'])
                state = json.load(open(startup['path'], 'r'))
                print('[+] loading checkpoint:', state['checkpoint_path'])
                last_checkpoint = os.path.dirname(state['checkpoint_path'])
                saver.restore(sess, tf.train.latest_checkpoint(last_checkpoint))
                iteration = state['iteration']

                best_loss = state['best_loss']
                if 'train_loss' in state:
                    best_loss = state['best_loss']

                checkpoint_path = state['checkpoint_path']
                # log_string = 'next-' + state['log_string']
            except:
                print('[!] no models to checkpoint from..')
                raise
            writer = tf.summary.FileWriter(log_string)

            seq = iaa.Sequential(
                iaa.OneOf([
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
                ]),
            )

            from collections import defaultdict
            
            EXAMPLES_CACHE_LIMIT = 512
            HARD_INJECT = 0.0
            hard_examples = defaultdict(list)
            
            def get_data_from_tip(tip, batch_size):
                features = []
                labels = []
                descriptions = []
                hard_to_inject = HARD_INJECT * batch_size
                hard_to_inject -= hard_to_inject % 2
                # batch_size -= hard_to_inject
                injected = 0
                samples_avail = 0
                for key, samples in hard_examples.items():
                    if len(samples) < int(hard_to_inject / 2):
                        continue
                    samples_avail += len(samples)
                    for desc, sample, label in random.sample(samples, int(hard_to_inject / 2)):
                        labels.append(label)
                        descriptions.append(desc)
                        features.append(sample)
                        injected += 1

                if injected > 0:
                    print('[+] {} samples were injected! ({} avail)'.format(injected, samples_avail))

                batch_size -= injected
                for i in range(batch_size):
                    data = tip.get()
                    d, f, l = data
                    features.append(f.reshape((224, 224, 1)))
                    labels.append(l)
                    descriptions.append(d)

                return descriptions, np.array(features), np.array(labels)

            def mix(a, b, mix_factor):
                return a * (1.0 - mix_factor) + b * mix_factor

            # from tensorflow.python import debug as tf_debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
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
                descriptions, features, labels = get_data_from_tip(tip, batch_size)
                iteration += 1
                # import pdb; pdb.set_trace()
                mean = 0.172840994091
                std = 0.206961060284
                # features = tf.map_fn(lambda f: tf.image.per_image_standardization(f), features)
                # try:
                #     with np.errstate(all='raise'):
                #         normalized = (features - mean) / std
                #         features = normalized
                # except Exception:
                #     print("[!] Warning detected normalizing, skipping..")
                #     tb = traceback.format_exc()
                #     open("numpy_warns.log", 'ab').write(str(descriptions).encode('utf-8'))
                #     open("numpy_warns.log", 'ab').write(str(tb).encode('utf-8'))
                #     open("numpy_warns.log", 'a').write('------------------------------------')
                    # import pdb; pdb.set_trace()
                    # continue
                merged_summaries = light_summary
                if iteration % heavy_summary_steps == 0:
                    merged_summaries = heavy_summary
                # mix_factor = min(float(iteration), steps_to_decay) / steps_to_decay
                # learning_rate = mix(start_learning_rate, end_learning_rate, mix_factor)

                # #################
                #   AUGMENTATION
                # #################
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

                # for i, pred in enumerate(_corr_pred):
                #     if pred:
                #         continue
                #     x = features[i]
                #     y = labels[i].tolist()
                #     desc = descriptions[i]
                #     key = np.argmax(y)
                #     EXAMPLES_CACHE_LIMIT = 8192
                #     samples = hard_examples[key]
                #     samples.append((desc, x, y))
                #     while len(samples) > EXAMPLES_CACHE_LIMIT:
                #         samples.pop(0)

                    
                # if val_acc < 0.96:
                #     with open('low_descriptions.log', 'ab') as f:
                #         f.write(b'[+] low rate: acc:%f loss:%f\n' % (val_acc, train_loss))
                #         for d in descriptions:
                #             f.write(b'\t%s\n' % d.encode('utf-8'))
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