from six.moves import xrange
import better_exceptions
import datetime
from functools import partial
import os

import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import shutil
import traceback
import warnings as w
w.simplefilter(action='ignore', category=FutureWarning)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class NomixModel:

    def __init__(self, params, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    @property
    def loss(self):
        raise NotImplemented()

    @property
    def accuracy(self):
        raise NotImplemented()

    @property
    def prediction(self):
        raise NotImplemented()

    @classmethod
    def create_optimizer(cls, params):
        if 'learning_rate' not in params:
            raise RuntimeError()
        
        learning_rate = params['learning_rate']
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.96, staircase=True)
        return tf.train.AdamOptimizer(learning_rate)

    @classmethod
    def create_inputs(cls, splits):
        x_mixed = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='x_mixed')
        y_src1 = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='y_src1')
        y_src2 = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='y_src2')
    
    @classmethod
    def create_outputs(cls, splits):
        raise NotImplementedError


def train(name, devices, tip, params, model_cls):
    IM_HEIGHT = audio.MELS
    IM_WIDTH = audio.MELS
    IM_CHANNEL = 1
    batch_size = 256
    iteration = 0
    min_loss = 1e10
    params_string = 'rl{rnn_layers}-hl{hidden_size}-sl{seq_len}-bs{batch_size}-lr{learning_rate:1.0e}'.format(**params)
    checkpoint_base = 'D:\\checkpoint_rnn'
    startup_file_name = 'startup.rnn.{}.json'.format(params_string)
    models_to_keep = 3
    models_history = []
    summary_steps = 10
    im_summary_steps = 25
    checkpoint_steps = 250
    model_version = 2
    learning_rate = 1e-3
    num_classes = 2
    time_str = datetime.datetime.now().strftime('%m-%d--%H%M%S')
    log_string = 'logs/rnn-{}/{}'.format(model_version, params_string)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    x, x_split = model_cls.create_inputs(splits=len(devices))
    y, y_split = model_cls.create_outputs(splits=len(devices))
    keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob')
    global_step = tf.Variable(0, trainable=False)
    summary = []
    
    x_split = tf.split(x_mixed, len(devices))
    y1_split = tf.split(y_src1, len(devices))
    y2_split = tf.split(y_src2, len(devices))

    writer = tf.summary.FileWriter(log_string)

    all_loss = []
    all_accuracy = []
    all_prediction = []
    all_gradient = []
    im_summaries = tf.summary.merge([s1, s2])

    optimizer = model_cls.create_optimizer(params)
    models = []
    summary = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(devices)):
            with tf.device(devices[i]):
                model = model_cls(i, params, x_split[i], y_split[i])
                models.append(model)
                tf.get_variable_scope().reuse_variables()
                loss = model.loss
                all_loss.append(loss)
                grad = optimizer.compute_gradients(loss)
                all_gradient.append(grad)
                accuracy = model.accuracy
                all_accuracy.append(accuracy)
                all_prediction.append(model.prediction)
            with tf.name_scope('tower_%d' % i) as scope:
                summary.append(tf.summary.scalar('loss', loss))
                summary.append(tf.summary.scalar('accuracy', accuracy))
                grad_norm = tf.norm(grad[0][0])
                summary.append(tf.summary.scalar('gradient', grad_norm))

    vars_ = tf.trainable_variables()
    loss = tf.reduce_mean(all_loss)
    accuracy = tf.reduce_mean(all_accuracy)
    grad_avg = average_gradients(all_gradient)
    train_op = optimizer.apply_gradients(grad_avg, global_step=global_step)
    
    with tf.name_scope('all'):
        summary.append(tf.summary.scalar('loss', loss))
        grad_norm = tf.norm(grads_[0][0])
        summary.append(tf.summary.scalar('gradient', grad_norm))
        summary.append(tf.summary.scalar('accuracy', accuracy))
    summary = tf.summary.merge(summary)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess.run(init_op)

    for model in models:
        model.load(sess)

    while True:
        descriptions, features, labels1, labels2 = get_data_from_tip(tip, batch_size)
        features = features.reshape((batch_size, IM_HEIGHT, IM_WIDTH))
        labels1 = labels1.reshape((batch_size, IM_HEIGHT, IM_WIDTH))
        labels2 = labels2.reshape((batch_size, IM_HEIGHT, IM_WIDTH))

        # import pdb; pdb.set_trace()
        to_run = [train_op, summary, loss, global_step, accuracy]
        feed_dict = {
            x_mixed: spec_to_batch(features),
            y_src1: spec_to_batch(labels1),
            y_src2: spec_to_batch(labels2),
            keep_prob: 0.5
        }
        _, _summary, _loss, iteration, train_acc = sess.run(to_run, feed_dict)
        min_loss = min(min_loss, _loss)
        log_str = '[+] {} Iteration {:10}, loss {:>15.5f}, min_loss {:>15.5f}, accuracy {:>9.5f}%'
        print(log_str.format(test_num, iteration, _loss, min_loss, train_acc))
        if iteration % summary_steps == 0:
            writer.add_summary(_summary, iteration)

        if iteration % im_summary_steps == 0:
            to_run = all_logits[0]
            feed_dict = {
                x_mixed: spec_to_batch(np.array([control_img] * len(devices))),
                keep_prob: 1.0
            }
            out_control_y1, out_control_y2 = sess.run(to_run, feed_dict)
            
            out_control_y1 = batch_to_spec(out_control_y1, 1)[:,:,:224].reshape((224, 224))
            out_control_y2 = batch_to_spec(out_control_y2, 1)[:,:,:224].reshape((224, 224))

            cm_hot = mpl.cm.get_cmap('hot')
            # import pdb; pdb.set_trace()
            out_control_y1 = np.uint8(cm_hot(out_control_y1) * 255).reshape((1, 224, 224, 4))
            out_control_y2 = np.uint8(cm_hot(out_control_y2) * 255).reshape((1, 224, 224, 4))

            feed_dict = {
                control_y1: out_control_y1,
                control_y2: out_control_y2
            }
            s = sess.run(im_summaries, feed_dict)
            writer.add_summary(s, iteration)

        if iteration % checkpoint_steps == 0 and should_save:
            if os.path.isfile('nomix_pdb'):
                import pdb
                pdb.set_trace()
            # print('\t\tNew Best Loss!')
            # best_loss = train_loss
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            
            checkpoint_dir = os.path.join(checkpoint_base, '{}-{}'.format(test_num, timestamp))
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            print('\t\tSaving model to:' + checkpoint_path)
            saver.save(sess, checkpoint_path, global_step=global_step)
            state = {
                'iteration': int(iteration),
                # 'val_acc': float(val_acc),
                'min_loss': int(min_loss),
                'checkpoint_path': checkpoint_path,
                'log_string': log_string,
                'params': params,
            }
            state_path = os.path.join(checkpoint_dir, 'state.json')
            open(state_path, 'w').write(json.dumps(state))
            startup = {
                'path': state_path,
            }
            open(startup_file_name, 'w').write(json.dumps(startup))
            models_history.append(checkpoint_dir)
            while len(models_history) > models_to_keep:
                try:
                    path_to_del = models_history.pop(0)
                    print('[+] deleting model', path_to_del)
                    shutil.rmtree(path_to_del)
                except:
                    print('[+] failed to delete')
                    traceback.print_exc()

        if iteration >= params['max_iteration']:
            break

    print('[+] done!')
    # iteration += 1

def train_parallel(devices, tips):
    load_prev = False
    should_save = True
    tests = []
    # for bs in [32, 64, 128, 256, 512]:
    #     for rl in [3, 2]:
    #         for lr in [1e-3, 1e-2, 1e-4, 1e-5]:
    #             for sl in [4, 8, 12, 16, 20, 24]:
    #                 for hs in [256, 512, 1024]:
    #                     tests.append(dict(
    #                         max_iteration=4000,
    #                         seq_len=sl,
    #                         hidden_size=hs,
    #                         rnn_layers=rl,
    #                         batch_size=bs,
    #                         learning_rate=lr
    #                     ))
    tests.append(dict(
                max_iteration=1000000,
                seq_len=4,
                hidden_size=512,
                rnn_layers=3,
                batch_size=128,
                learning_rate=1e-3
            ))
    tests.append(dict(
                max_iteration=1000000,
                seq_len=16,
                hidden_size=512,
                rnn_layers=3,
                batch_size=128,
                learning_rate=1e-3
            ))
    i = 0
    # tests.pop(0)
    # tests.pop(0)
    import pprint
    while tests:
        try:
            test1 = tests.pop(0)
            print('[+] going with the following test:')
            pprint.pprint(test1)
            t1 = Thread(target=train, args=(i, ['/gpu:0'], tips[0], test1, load_prev, should_save))
            t1.daemon = True
            t1.start()
            if not tests:
                t1.join()
                continue
            i += 1
            test2 = tests.pop(0)
            print('[+] going with the following test:')
            pprint.pprint(test2)
            t2 = Thread(target=train, args=(i, ['/gpu:1'], tips[1], test2, load_prev, should_save))
            t2.daemon = True
            t2.start()

            try:
                t1.join()
                t2.join()
            except KeyboardInterrupt:
                print('[+] keyint 1.. skipping')
                t1.join()
                t2.join()
                pass
            i += 1
        except KeyboardInterrupt:
            print('[+] keyint 2')
        except:
            raise

if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
    import numpy as np
    from pipeline import Pipeline
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap
    from commons.ops import *
    import audio
    # from multiprocessing.pool import ThreadPool
    from threading import Thread
    with open('configs/win_client.rnn.yaml') as f:
        config = yaml.safe_load(f)
    amp = Pipeline(config['pipeline'])
    amp.run()
    amp.keepalive(block=False)
    tip = amp._get_tip_queue()
    tip2 = amp._stages[-1].output_queue2
    tips = [tip, tip]
    devices = ['/gpu:0', '/gpu:1']
    
    load_prev = True
    should_save = True
    params = dict(
        max_iteration=10000000,
        hidden_size=1024,
        rnn_layers=6,
        batch_size=256,
        learning_rate=1e-3,
        seq_len=16,
    )

    # train(0, devices, tip, params, load_prev=load_prev, should_save=should_save)
    train_parallel(devices, tips)
    print('[+] done all')
        