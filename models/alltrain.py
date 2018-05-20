import os
import time
import math

import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def

from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS
import numpy as np
import time
import random
import pickle
import datetime
import vgg16
import cache_file
import json

from imgaug import augmenters as iaa

default_device = '/gpu:0'
# default_device = '/cpu:0'

num_hidden_neurons = 256

vgg_mean = [103.939, 116.779, 123.68]
#model_version = int(time.time())
model_version = 1
model_path = 'models/model-{}/'.format(model_version)

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

accuracy_print_steps = 100
model_version = 1
model_path = 'models/model-{}/'.format(model_version)
batch_size = 40
learning_rate = 0.0000001
num_hidden_layers = 2
hidden_layer_size = 512
keep_prob = 1.0
time_str = datetime.datetime.now().strftime('%m-%d--%H%M%S')
training = True

def train():
    with tf.device(default_device):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            with tf.name_scope("inputs"):
                # _images = tf.placeholder(tf.float32, shape=(None, 4096), name='images')
                _images = tf.placeholder(tf.float32, [None, 224, 224, 1])
                _is_training = tf.placeholder(tf.bool, name='is_training')
                _keep_prob = tf.placeholder(tf.float32, name='keep_probability')
            # imgs = tf.placeholder(tf.float32, [None, 224, 224, 1])
            model = vgg16.Vgg16(_images, '../vgg16_weights.npz', classes=2, mean=[0.343388929118], trainable=training)

            with tf.name_scope("targets"):
                _labels = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

            with tf.name_scope("outputs"):
                output_weights = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(hidden_layer_size, 2), mean=0.0, stddev=0.01),
                    dtype=tf.float32, name="output_weights"
                )

                output_bias = tf.Variable(initial_value=tf.zeros(2), dtype=tf.float32, name="output_bias")

                logits = model.fc3l
                predictions = tf.nn.softmax(logits, name='predictions')

                tf.summary.histogram("output_weights", output_weights)
                tf.summary.histogram("output_bias", output_bias)
                tf.summary.histogram("predictions", predictions)

            with tf.name_scope("cost"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=_labels, name='cross_entropy')
                cost = tf.reduce_mean(cross_entropy, name='cost')

                tf.summary.scalar("cost", cost)

            with tf.name_scope("train"):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):         
                    starter_learning_rate = learning_rate
                    # global_step = tf.Variable(0, trainable=False)
                    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                    #                                         100000, 0.96, staircase=True)       
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
                    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(_labels, 1), name='correct_predictions')
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

                    tf.summary.scalar("accuracy", accuracy)

            merged_summaries = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            
            iteration = 0
            best_loss = 9999999999
            batch_num = 0
            best_acc = 0
            # test_batches = get_batches('TEST', batch_size=batch_size, repeat=True)

            log_string = 'logs/{}/{}'.format(model_version, time_str)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            try:
                print('[+] loading startup.json')
                startup = json.load(open('startup.json', 'r'))
                print('[+] loading path:', startup['path'])
                state = json.load(open(startup['path'], 'r'))
                print('[+] loading checkpoint:', state['checkpoint_path'])
                saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(state['checkpoint_path'])))
                # iteration = state['iteration']
                # best_acc = state['best_acc']
                # best_loss = state['best_loss']
                best_acc = state['best_acc']
                if 'val_acc' in state:
                    best_acc = state['val_acc']

                best_loss = state['best_loss']
                if 'train_loss' in state:
                    best_loss = state['train_loss']
                
                checkpoint_path = state['checkpoint_path']
                # log_string = 'next-' + state['log_string']
            except:
                print('[!] no models to checkpoint from..')
            writer = tf.summary.FileWriter(log_string)
            # import numpy as np
            # vcol = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # for v in vcol:
            #     res = sess.run([v])
            #     np.save(v.name.replace('/', '_').replace(':', '_')+'.npy', res)
            
            seq = iaa.Sequential(
                iaa.OneOf([
                            iaa.GaussianBlur((0.45)),  # blur images with a sigma between 0 and 3.0
                            iaa.Add((-90.0, 90.0), per_channel=False),
                            iaa.Multiply((0.5, 1.5), per_channel=False),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.07*255, 0.07*255), per_channel=False),
                            iaa.Dropout(p=0.07, per_channel=False),
                            iaa.CoarseDropout(p=0.05, size_percent=(0.2, 0.9),  per_channel=False),
                            iaa.SaltAndPepper(p=0.07, per_channel=False),
                            iaa.Salt(p=0.07, per_channel=False),
                            iaa.Pepper(p=0.07, per_channel=False),
                            iaa.ContrastNormalization(alpha=(1.2, 1.5)),
                            iaa.ElasticTransformation(alpha=(0.7)),
                        ]),
            )

            cc = cache_file.CacheCollection({'filename': 'T:\\cache\\AudioToImage', 'seek_policy': 'ONE_SHOT', 'max_size': 2147483648, 'max_split': 50})
            while True:
                train = cc.random_iterator(batch_size, test=False)
                test = cc.random_iterator(batch_size*2, test=True)
                # train_dataset.shuffle()
                # test_dataset.shuffle()
                # test_batches = test_dataset.get_batches(batch_size)
                # for batch_train_images, batch_train_labels in train_dataset.get_batches(batch_size):
                for features, labels in train:
                    train_loss, _, p, summary = sess.run(
                        [cost, optimizer, logits, merged_summaries],
                        feed_dict={
                            _images: seq.augment_images(features * 255) / 255,
                            _labels: labels,
                            _keep_prob: keep_prob,
                            _is_training: training
                        })

                    iteration += 1
                    print('[+] iteration {}'.format(iteration))

                    if iteration % accuracy_print_steps == 0:
                        if not writer == None:
                            writer.add_summary(summary, iteration)

                        val_features, val_labels = next(test)

                        val_acc, val_summary = sess.run([accuracy, merged_summaries], feed_dict ={
                            _images: seq.augment_images(val_features * 255) / 255,
                            _labels: val_labels,
                            _keep_prob: 1.,
                            _is_training: False
                        })


                        print('\tIteration {} Accuracy: {} Loss: {}'.format(iteration, val_acc, train_loss))
                        print('\t\t Best Accuracy: {} Best Loss: {}'.format(iteration, best_acc, best_loss))
                        if val_acc >= best_acc or train_loss <= best_loss:
                            if train_loss <= best_loss:
                                best_loss = train_loss
                            if val_acc >= best_acc:
                                best_acc = val_acc
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                            checkpoint_path = os.path.join('save', timestamp, 'model.ckpt')
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
                            state_path = os.path.join('save', timestamp, 'state.json')
                            open(state_path, 'w').write(json.dumps(state))
                            startup = {
                                'path': state_path,
                            }
                            open('startup.json', 'w').write(json.dumps(startup))

                    batch_num += 1
            if saved_model_path:
                ### Save graph and trained variables
                builder = saved_model_builder.SavedModelBuilder(saved_model_path)
                builder.add_meta_graph_and_variables(
                    sess, [SERVING],
                    signature_def_map = {
                        DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def(
                            inputs = { PREDICT_INPUTS: _images },
                            outputs = { PREDICT_OUTPUTS: predictions }
                        )
                    }
                )

                builder.save()


train()
# cc = cache_file.CacheCollection({'filename': 'T:\\cache\\AudioToImage', 'seek_policy': 'ONE_SHOT', 'max_size': 2147483648, 'max_split': 50})
# print(cc.mean())