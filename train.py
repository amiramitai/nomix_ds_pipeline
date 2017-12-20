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

from pipeline import AudioEncoderPipeline

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

def train(writer, hidden_layer_size, learning_rate, num_hidden=1, keep_prob=0.5, batch_size=64, training=True, saved_model_path=None):
    with tf.device(default_device):
        with tf.Session(graph=tf.Graph()) as sess:

            with tf.name_scope("inputs"):
                _images = tf.placeholder(tf.float32, shape=(None, 4096), name='images')
                _is_training = tf.placeholder(tf.bool, name='is_training')
                _keep_prob = tf.placeholder(tf.float32, name='keep_probability')

            with tf.name_scope("targets"):
                _labels = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

            prev_size = 4096
            next_input = _images
            
            for i in range(num_hidden):
                with tf.variable_scope("hidden_layer_{}".format(i)):
                    hidden_weights = tf.Variable(
                        initial_value = tf.truncated_normal([prev_size, hidden_layer_size], mean=0.0, stddev=0.01),
                        dtype=tf.float32, name="hidden_weights"
                    )

                    hidden_bias = tf.Variable(
                        initial_value = tf.zeros(hidden_layer_size), 
                        dtype=tf.float32,
                        name="hidden_bias"
                    )

                    hidden = tf.matmul(next_input, hidden_weights) + hidden_bias
                    # hidden = tf.layers.batch_normalization(hidden, training=_is_training)
                    hidden = tf.nn.relu(hidden, name="hidden_relu")
                    hidden = tf.nn.dropout(hidden, keep_prob=_keep_prob, name='hidden_dropout')

                    tf.summary.histogram("hidden_weights_{}".format(i), hidden_weights)
                    tf.summary.histogram("hidden_bias_{}".format(i), hidden_bias)
                    
                    next_input = hidden
                    prev_size = hidden_layer_size


            with tf.name_scope("outputs"):
                output_weights = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(hidden_layer_size, 2), mean=0.0, stddev=0.01),
                    dtype=tf.float32, name="output_weights"
                )

                output_bias = tf.Variable(initial_value=tf.zeros(2), dtype=tf.float32, name="output_bias")

                logits = tf.matmul(next_input, output_weights) + output_bias
                predictions = tf.nn.softmax(logits, name='predictions')

                tf.summary.histogram("output_weights", output_weights)
                tf.summary.histogram("output_bias", output_bias)
                tf.summary.histogram("predictions", predictions)

            with tf.name_scope("cost"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_labels, name='cross_entropy')
                cost = tf.reduce_mean(cross_entropy, name='cost')

                tf.summary.scalar("cost", cost)

            with tf.name_scope("train"):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):                
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
                    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(_labels, 1), name='correct_predictions')
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

            ### merge summaries
            merged_summaries = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())
            
            iteration = 0
            # test_batches = get_batches('TEST', batch_size=batch_size, repeat=True)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            # try:
            #     saver.restore(sess, tf.train.latest_checkpoint('./save/2017-12-08_215623'))
            # except:
            #     print('[!] no models to checkpoint from..')
            # import numpy as np
            # vcol = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # for v in vcol:
            #     res = sess.run([v])
            #     np.save(v.name.replace('/', '_').replace(':', '_')+'.npy', res)
            best_acc = 0
            best_loss = 1
            batch_num = 0            
            while True:
                train_dataset = AudioEncoderPipeline()
                test_dataset = train_dataset
                # train_dataset.shuffle()
                # test_dataset.shuffle()
                # test_batches = test_dataset.get_batches(batch_size)
                # for batch_train_images, batch_train_labels in train_dataset.get_batches(batch_size):
                for batch_train_images, batch_train_labels in train_dataset.iterate(batch_size):
                    train_loss, _, p, summary = sess.run(
                        [cost, optimizer, logits, merged_summaries],
                        feed_dict={
                            _images: np.array(batch_train_images),
                            _labels: batch_train_labels,
                            _keep_prob: keep_prob,
                            _is_training: training
                        })

                    iteration += 1
                    print('[+] iteration {}'.format(iteration))

                    if iteration % accuracy_print_steps == 0:
                        if not writer == None:
                            writer.add_summary(summary, iteration)

                        val_images, val_labels = next(test_batches)

                        val_acc, val_summary = sess.run([accuracy, merged_summaries], feed_dict ={
                            _images: val_images,
                            _labels: val_labels,
                            _keep_prob: 1.,
                            _is_training: False
                        })


                        print('\tIteration {} Accuracy: {} Loss: {}'.format(iteration, val_acc, train_loss))
                        if val_acc >= best_acc:
                            if train_loss <= best_loss:
                                best_acc = val_acc
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                                checkpoint_path = os.path.join('save', timestamp, 'model.ckpt')
                                print('\t\tSaving model to:' + checkpoint_path)
                                saver.save(sess, checkpoint_path, global_step=batch_num)

                    batch_num += 1
            if not saved_model_path == None:
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


batch_size = 12
learning_rate = 0.000005
num_hidden_layers = 2
hidden_layer_size = 512
keep_prob = 1.0
log_string = 'logs/{}/lr={},hl={},hs={},kp={},bs={}'.format(model_version, learning_rate, num_hidden_layers, hidden_layer_size, keep_prob, batch_size)
writer = tf.summary.FileWriter(log_string)
print("\n\nStarting {}".format(log_string))
train(writer, hidden_layer_size, learning_rate, num_hidden_layers, keep_prob, batch_size, training=True, saved_model_path=model_path)