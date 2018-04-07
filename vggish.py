########################################################################################
# Davi Frossard, 2016                                                                  #
# Vggish implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
# from imagenet_classes import class_names
import sys
import os
import pickle
import traceback
import math
from tensorflow.python import pywrap_tensorflow


def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]
    # x2: Tensor("conv1_1/transpose:0", shape=(64, 5, 5, 1), dtype=float32)
    # x3: Tensor("conv1_1/Reshape:0", shape=(8, 40, 5, 1), dtype=float32)
    # x4: Tensor("conv1_1/transpose_1:0", shape=(8, 5, 40, 1), dtype=float32)
    # x5: Tensor("conv1_1/Reshape_1:0", shape=(1, 40, 40, 1), dtype=float32)
    # x6: Tensor("conv1_1/transpose_2:0", shape=(40, 40, 1, 1), dtype=float32)
    # x7: Tensor("conv1_1/transpose_3:0", shape=(1, 40, 40, 1), dtype=float32)
    

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels])) #3
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    # return tf.image.convert_image_dtype(x7, dtype = tf.uint8) 
    return x7


conv_vars = [
    'BatchNorm_1/beta',
    'BatchNorm_1/moving_mean',
    'BatchNorm_1/moving_variance',
    'conv1_1/biases',
    'conv1_1/weights',
    'BatchNorm_2/beta',
    'BatchNorm_2/moving_mean',
    'BatchNorm_2/moving_variance',
    'conv2_1/biases',
    'conv2_1/weights',
    'BatchNorm_3/beta',
    'BatchNorm_3/moving_mean',
    'BatchNorm_3/moving_variance',
    'conv3_1/biases',
    'conv3_1/weights',
    'BatchNorm_4/beta',
    'BatchNorm_4/moving_mean',
    'BatchNorm_4/moving_variance',
    'conv3_2/biases',
    'conv3_2/weights',
    'BatchNorm_5/beta',
    'BatchNorm_5/moving_mean',
    'BatchNorm_5/moving_variance',
    'conv4_1/biases',
    'conv4_1/weights',
    'BatchNorm_6/beta',
    'BatchNorm_6/moving_mean',
    'BatchNorm_6/moving_variance',
    'conv4_2/biases',
    'conv4_2/weights',
    'BatchNorm_7/beta',
    'BatchNorm_7/moving_mean',
    'BatchNorm_7/moving_variance',
]

fc_vars = [
    'fc1/biases',
    'fc1/weights',
    'BatchNorm_8/beta',
    'BatchNorm_8/moving_mean',
    'BatchNorm_8/moving_variance',
    'fc2/biases',
    'fc2/weights',
    'BatchNorm/beta',
    'BatchNorm/moving_mean',
    'BatchNorm/moving_variance',
    'fc3/biases',
    'fc3/weights',
]


class Vggish:
    def __init__(self, imgs, classes=2, trainable=False, batch_norm=True,
                 dropout=None, only_dense=False, dense_size=64, bn_trainable=False):
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.classes = classes
        self.only_dense = only_dense
        self.imgs = imgs
        self.trainable = trainable
        self.bn_trainable = bn_trainable
        self.parameters = []
        self.summaries = []
        self.dense_size = dense_size
        self.heavy_summaries = []
        # self.weight_init_type = "msra"
        self.weight_init_type = "truc"
        self.w_initializer = None
        self.convlayers()
        self.fc_layers()

    def get_vars_to_train(self):
        vars_to_train = []
        for v in tf.trainable_variables():
            if v is None:
                continue
            
            if self.bn_trainable:
                if 'BatchNorm' in v.name:
                    print('[+] skipping:', v.name)
                    continue
            
            vars_to_train.append(v)
        return vars_to_train

    def add_variable_summaries(self, var, name='light'):
        if not self.trainable:
            return
        
        summaries = self.summaries
        if name == 'heavy':
            summaries = self.heavy_summaries
        
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            summaries.append(tf.summary.scalar('mean', mean))
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            summaries.append(tf.summary.scalar('stddev', stddev))
            summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))
            summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))
            # summaries.append(tf.summary.histogram('histogram', var))

    def add_weights_summary(self, kernel, shape, pad=1):
        if not self.trainable:
            return
        
        x = put_kernels_on_grid(kernel, 8, 8)
        summary = tf.summary.image('conv1/features', x)
        self.heavy_summaries.append(summary)        

    def init_weight(self, shape, stddev=1e-01):
        if self.weight_init_type == "msra":
            if self.w_initializer is None:
                self.w_initializer = tf.contrib.layers.variance_scaling_initializer()
            return self.w_initializer(shape)

        if self.w_initializer is None:
            self.w_initializer = tf.truncated_normal

        return self.w_initializer(shape, stddev=stddev)

    def convlayers(self):
        init_stddev = 0.1
        trainable = self.trainable
        if self.only_dense:
            trainable = False
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            images = self.imgs
            if self.batch_norm:
                images = tf.contrib.layers.batch_norm(images)
        # reader = pywrap_tensorflow.NewCheckpointReader("S:\\Projects\\nomix\\vggish_model.ckpt")
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            # stddev = np.sqrt(2 / np.prod(images.get_shape().as_list()[1:]))
            stddev = 1e-01
            kernel = tf.Variable(self.init_weight([3, 3, 1, 64], stddev=stddev),
                                 trainable=trainable, name='weights')
            self.add_variable_summaries(kernel)
            self.add_weights_summary(kernel, [3, 3, 1, 64])
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            # biases = tf.Variable(reader.get_tensor("vggish/conv1/biases"), dtype=tf.float32, name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.conv1_1 = out
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            # stddev = np.sqrt(2 / np.prod(self.pool1.get_shape().as_list()[1:]))
            stddev = 1e-01
            kernel = tf.Variable(self.init_weight([3, 3, 64, 128], stddev=stddev),
                                 trainable=trainable, name='weights')
            self.add_variable_summaries(kernel)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.conv2_1 = out
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        if self.dropout:
            self.pool2 = tf.nn.dropout(self.pool2, 0.3)


        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            # stddev = np.sqrt(2 / np.prod(self.pool2.get_shape().as_list()[1:]))
            stddev = 1e-01
            kernel = tf.Variable(self.init_weight([3, 3, 128, 256], stddev=stddev),
                                 trainable=trainable, name='weights')
            self.add_variable_summaries(kernel)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.conv3_1 = out
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            # stddev = np.sqrt(2 / np.prod(self.conv3_1.get_shape().as_list()[1:]))
            stddev = 1e-01
            kernel = tf.Variable(self.init_weight([3, 3, 256, 256], stddev=stddev),
                                 trainable=trainable, name='weights')
            self.add_variable_summaries(kernel)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.conv3_2 = out
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        if self.dropout:
            self.pool3 = tf.nn.dropout(self.pool3, 0.3)

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            # stddev = np.sqrt(2 / np.prod(self.pool3.get_shape().as_list()[1:]))
            stddev = 1e-01
            kernel = tf.Variable(self.init_weight([3, 3, 256, 512], stddev=stddev),
                                 trainable=trainable, name='weights')
            self.add_variable_summaries(kernel)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.conv4_1 = out
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            # stddev = np.sqrt(2 / np.prod(self.conv4_1.get_shape().as_list()[1:]))
            stddev = 1e-01
            kernel = tf.Variable(self.init_weight([3, 3, 512, 512], stddev=stddev),
                                 trainable=trainable, name='weights')
            self.add_variable_summaries(kernel)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=trainable, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.conv4_2 = out
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        if self.dropout:
            self.pool4 = tf.nn.dropout(self.pool4, 0.3)

    def fc_layers(self):
        # fc1
        DENSE_SIZE = self.dense_size
        summary_name = 'light'
        with tf.name_scope('fc1') as scope:
            # stddev = np.sqrt(2 / np.prod(self.pool4.get_shape().as_list()[1:]))
            stddev = 1e-03
            shape = int(np.prod(self.pool4.get_shape()[1:]))
            fc1w = tf.Variable(self.init_weight([shape, DENSE_SIZE], stddev=stddev), name='weights')
            self.add_variable_summaries(fc1w, name=summary_name)
            fc1b = tf.Variable(tf.constant(1.0, shape=[DENSE_SIZE], dtype=tf.float32),
                               trainable=self.trainable, name='biases')
            pool4_flat = tf.reshape(self.pool4, [-1, shape])
            out = tf.nn.bias_add(tf.matmul(pool4_flat, fc1w), fc1b)
            out = tf.nn.relu(out)
            # out = tf.nn.leaky_relu(out)
            if self.dropout:
                out = tf.nn.dropout(out, 0.3)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.fc1 = out
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            # stddev = np.sqrt(2 / np.prod(self.fc1.get_shape().as_list()[1:]))
            stddev = 1e-03
            fc2w = tf.Variable(self.init_weight([DENSE_SIZE, DENSE_SIZE], stddev=stddev), name='weights')
            self.add_variable_summaries(fc2w, name=summary_name)
            fc2b = tf.Variable(tf.constant(1.0, shape=[DENSE_SIZE], dtype=tf.float32),
                               trainable=self.trainable, name='biases')
            out = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            out = tf.nn.relu(out)
            # out = tf.nn.leaky_relu(out)
            if self.dropout:
                out = tf.nn.dropout(out, 0.5)
            if self.batch_norm:
                out = tf.contrib.layers.batch_norm(out, trainable=self.bn_trainable)
            self.fc2 = out
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            # stddev = np.sqrt(2 / np.prod(self.fc2.get_shape().as_list()[1:]))
            stddev = 1e-01
            fc3w = tf.Variable(self.init_weight([DENSE_SIZE, self.classes], stddev=stddev), name='weights')
            self.add_variable_summaries(fc3w, name=summary_name)
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.classes], dtype=tf.float32),
                               trainable=self.trainable, name='biases')
            out = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            # if self.batch_norm:
            #     out = tf.contrib.layers.batch_norm(out)
            self.fc3l = out
            self.parameters += [fc3w, fc3b]
    
    def load_weights(self, weights_path, vars_names=None):
        if not vars_names:
            all_vars = conv_vars + fc_vars
            vars_names = all_vars
        
        def should_load(name):
            for tl in vars_names:
                if tl in name:
                    return True

            return False
        
        # reader = pywrap_tensorflow.NewCheckpointReader(latest)
        latest = tf.train.latest_checkpoint(weights_path)
        to_load = []
        for v in tf.global_variables():
            if not should_load(v.name):
                continue
            to_load.append(v)
        saver = tf.train.Saver(to_load, max_to_keep=None)
        saver.restore(tf.get_default_session(), latest)



if __name__ == '__main__':
    batch_size = 1
    default_device = '/cpu:0'
    weights_path = '/Users/amiramitai/Projects/nomix/2018-04-07_121235'
    with tf.device(default_device):
        with tf.Session(graph=tf.Graph(), config=tf.ConfigProto()) as sess:
            with tf.name_scope("inputs"):
                _images = tf.placeholder(tf.float32, [batch_size, 224, 224, 1])
                _is_training = tf.placeholder(tf.bool, name='is_training')
            model = Vggish(_images)
            model.load_weights(weights_path)
