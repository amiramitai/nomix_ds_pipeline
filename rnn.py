import warnings as w
w.simplefilter(action = 'ignore', category = FutureWarning)

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

SEQ_LEN = 4


class RNNModel:
    def __init__(self, x, y1, y2, keep_prob=None, n_rnn_layer=3, hidden_size=256, seq_len=4, training=True):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.seq_len = seq_len
        self.training = training
        self.keep_prob = keep_prob
        self.prob = 1.0
        if training:
            self.prob = 0.5
        # Network
        self.hidden_size = hidden_size
        self.n_layer = n_rnn_layer
        # self.net = tf.make_template('net', self._net)
        # self()
        self.logits = self._net()

    def __call__(self):
        return self.net()

    def _net(self):
        # RNN and dense layers
        # tf.contrib.rnn.NASCell(4*max_layers)
        # from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
        # cells = [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in range(self.n_layer)]
        cells = [tf.contrib.rnn.GRUCell(self.hidden_size) for _ in range(self.n_layer)]
        # cells = [tf.contrib.rnn.NASCell(self.hidden_size) for _ in range(self.n_layer)]
        rnn_layer = tf.contrib.rnn.MultiRNNCell(cells)
        if self.keep_prob is not None:
            rnn_layer = tf.contrib.rnn.DropoutWrapper(rnn_layer, output_keep_prob=self.keep_prob)
        with tf.variable_scope('RNN', initializer=tf.contrib.layers.xavier_initializer()):
            output_rnn, rnn_state = tf.nn.dynamic_rnn(rnn_layer, self.x, dtype=tf.float32)
            # import pdb; pdb.set_trace()
            # input_size = tf.shape(self.x)[2]
            input_size = self.x.shape[2].value
            y_hat_src1 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src1')
            y_hat_src2 = tf.layers.dense(inputs=output_rnn, units=input_size, activation=tf.nn.relu, name='y_hat_src2')

        # time-freq masking layer
        # y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x
        # y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x

        # src_mask1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x
        # src_mask2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x

        return y_hat_src1, y_hat_src2

    # def loss(self):
    #     pred_y_src1, pred_y_src2 = self.logits
    #     return tf.reduce_mean(tf.square(self.y1 - pred_y_src1) + tf.square(self.y2 - pred_y_src2), name='loss')
    
    def loss(self):
        y_hat_src1, y_hat_src2 = self.logits
        return tf.reduce_mean(tf.square(self.y1 - self.x*y_hat_src1) + tf.square(self.y2 - self.x*y_hat_src2), name='loss')

    # def acc(self):
    #     sample_shape = (-1, self.seq_len, audio.MELS)
    #     reshaped = tf.reshape(self.logits, (2, *sample_shape))
    #     pred_y_src1, pred_y_src2 = tf.split(reshaped, 2, axis=0)
    #     pred_y_src1 = tf.reshape(pred_y_src1, sample_shape)
    #     pred_y_src2 = tf.reshape(pred_y_src2, sample_shape)
    #     total_pixels = tf.cast(tf.multiply(tf.reduce_prod(tf.shape(reshaped)), 1), tf.float32)
    #     err_pixel_sum = tf.reduce_sum(tf.abs(self.y1 - pred_y_src1) + tf.abs(self.y2 - pred_y_src2))
    #     return tf.multiply(tf.subtract(1.0, tf.divide(err_pixel_sum, total_pixels)), 100.0, name='accuracy')
    
    def acc(self):
        sample_shape = (-1, self.seq_len, audio.MELS)
        reshaped = tf.reshape(self.logits, (2, *sample_shape))
        pred_y_src1, pred_y_src2 = tf.split(reshaped, 2, axis=0)
        pred_y_src1 = tf.reshape(pred_y_src1, sample_shape)
        pred_y_src2 = tf.reshape(pred_y_src2, sample_shape)
        total_pixels = tf.cast(tf.multiply(tf.reduce_prod(tf.shape(reshaped)), 1), tf.float32)
        err_pixel_sum = tf.reduce_sum(tf.abs(self.y1 - self.x*pred_y_src1) + tf.abs(self.y2 - self.x*pred_y_src2))
        return tf.multiply(tf.subtract(1.0, tf.divide(err_pixel_sum, total_pixels)), 100.0, name='accuracy')


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

#https://github.com/tensorflow/tensorflow/issues/13434
def train(test_num, devices, tip, params=None, load_prev=True, should_save=True):
    # with tf.variable_scope('params') as params:
    #     pass

    if not params:
        params = dict(
            max_iteration=10000000,
            hidden_size=256,
            rnn_layers=3,
            batch_size=256,
            learning_rate=1e-3,
            seq_len=4,
        )

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

    # im = tf.placeholder(tf.float32, [batch_size, IM_HEIGHT, IM_WIDTH, IM_CHANNEL])
    # 19 + unlabeled area(plus ignored labels)
    # gt = tf.placeholder(tf.float32, [batch_size, IM_HEIGHT, IM_WIDTH, 2])
    x_mixed = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='x_mixed')
    y_src1 = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='y_src1')
    y_src2 = tf.placeholder(tf.float32, shape=(None, None, audio.MELS), name='y_src2')
    keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob')
    global_step = tf.Variable(0, trainable=False)
    summary = []
    
    x_split = tf.split(x_mixed, len(devices))
    y1_split = tf.split(y_src1, len(devices))
    y2_split = tf.split(y_src2, len(devices))

    writer = tf.summary.FileWriter(log_string)

    all_logits = []
    all_loss = []
    all_acc = []
    all_preds = []
    eval_logits = []
    tower_grads = []
    control_y1 = tf.placeholder(tf.float32, shape=(1, audio.MELS, audio.MELS, 4), name='control_y1')
    control_y2 = tf.placeholder(tf.float32, shape=(1, audio.MELS, audio.MELS, 4), name='control_y2')
    s1 = tf.summary.image('s_control_y1', control_y1)
    s2 = tf.summary.image('s_control_y2', control_y2)
    im_summaries = tf.summary.merge([s1, s2])

    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.variable_scope(tf.get_variable_scope()):
        for i, (device, x, y1, y2) in enumerate(zip(devices, x_split, y1_split, y2_split)):
            with tf.device(device):
                with tf.name_scope('tower_%d' % i) as scope:
                    print(i, scope)
                    nrl = params['rnn_layers']
                    hs = params['hidden_size']
                    net = RNNModel(x, y1, y2, keep_prob, n_rnn_layer=nrl, hidden_size=hs, seq_len=params['seq_len'])
                    grads = optimizer.compute_gradients(net.loss())
                    tf.get_variable_scope().reuse_variables()
                    all_loss.append(net.loss())
                    all_acc.append(net.acc())
                    tower_grads.append(grads)
                    all_logits.append(net.logits)
    
    vars_ = tf.trainable_variables()    
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    # print('towers:', tower_grads)
    # import pdb; pdb.set_trace()
    loss = tf.reduce_mean(all_loss)
    acc = tf.reduce_mean(all_acc)
    # preds = tf.concat(all_preds, axis=0)
    # logits = tf.concat(eval_logits, axis=0)
    grads_ = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(grads_, global_step=global_step)

    summary = []
    for i, (loss_, acc_, grad) in enumerate(zip(all_loss, all_acc, tower_grads)):
        with tf.name_scope('tower_%d' % i):
            # summary.append(tf.summary.histogram('preds', preds_))
            summary.append(tf.summary.scalar('loss', loss_))
            
            # total_pixels = float(np.prod(split.get_shape()).value) / tf.shape(split)[0]
            # accuracy = (1.0 - (loss_ / total_pixels)) * 100.0
            # accuracy = tf.reduce_mean()
            summary.append(tf.summary.scalar('accuracy', acc_))
            grad_norm = tf.norm(grad[0][0])
            summary.append(tf.summary.scalar('gradient', grad_norm))

    with tf.name_scope('all'):
        summary.append(tf.summary.scalar('loss', loss))
        grad_norm = tf.norm(grads_[0][0])
        summary.append(tf.summary.scalar('gradient', grad_norm))
        # total_pixels = np.prod(gt.get_shape()).value / tf.shape(split)[0]
        accuracy = acc
        summary.append(tf.summary.scalar('accuracy', accuracy))
    summary = tf.summary.merge(summary)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess.run(init_op)
    
    try:
        if load_prev:
            print('[+] loading startup.json')
            startup = json.load(open('startup.rnn.json', 'r'))
            print('[+] loading path:', startup['path'])
            state = json.load(open(startup['path'], 'r'))
            print('[+] loading checkpoint:', state['checkpoint_path'])
            last_checkpoint = os.path.dirname(state['checkpoint_path'])

            to_load = []
            for v in tf.trainable_variables():
                to_load.append(v)
            s = tf.train.Saver(to_load, max_to_keep=None)

            latest = tf.train.latest_checkpoint(last_checkpoint)
            s.restore(sess, latest)
            
            # min_loss = state['min_loss']
            checkpoint_path = state['checkpoint_path']
            sess.run(tf.assign(global_step, state['iteration']))
    except:
        print('[!] no models to checkpoint from..')
        # raise
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    sess.graph.finalize()

    # augmenters = [
    #     # blur images with a sigma between 0 and 3.0
    #     iaa.Noop(),
    #     iaa.GaussianBlur(sigma=(0.5, 2.0)),
    #     iaa.Add((-50.0, 50.0), per_channel=False),
    #     iaa.AdditiveGaussianNoise(loc=0,
    #                                 scale=(0.07*255, 0.07*255),
    #                                 per_channel=False),
    #     iaa.Dropout(p=0.07, per_channel=False),
    #     iaa.CoarseDropout(p=(0.05, 0.15),
    #                         size_percent=(0.1, 0.9),
    #                         per_channel=False),
    #     iaa.SaltAndPepper(p=(0.05, 0.15), per_channel=False),
    #     iaa.Salt(p=(0.05, 0.15), per_channel=False),
    #     iaa.Pepper(p=(0.05, 0.15), per_channel=False),
    #     iaa.ContrastNormalization(alpha=(iap.Uniform(0.02, 0.03),
    #                                 iap.Uniform(1.7, 2.1))),
    #     iaa.ElasticTransformation(alpha=(0.5, 2.0)),
    # ]

    augmenters = [
        # blur images with a sigma between 0 and 3.0
        iaa.Noop(),
    ]

    seq = iaa.Sequential(iaa.OneOf(augmenters),)

    seq_len = params['seq_len']

    def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # Padding
        pad_len = 0
        if n_frames % seq_len > 0:
            pad_len = (seq_len - (n_frames % seq_len))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % seq_len == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, seq_len, freq))
        return batch

    def batch_to_spec(src, num_wav):
        # shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq, n_frames)
        batch_size, seq_len, freq = src.shape
        src = np.reshape(src, (num_wav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src
    
    def arr_and_to_batch(lst):
        import pdb; pdb.set_trace()
        return spec_to_batch(np.array(lst))

    def get_data_from_tip(tip, batch_size):
        features = []
        labels1 = []
        labels2 = []
        descriptions = []
        for i in range(batch_size):
            data = tip.get()
            d, f, l1, l2 = data
            features.append(f.reshape((IM_WIDTH, IM_HEIGHT, 1)))
            labels1.append(l1)
            labels2.append(l2)
            descriptions.append(d)
        return descriptions, np.array(features), np.array(labels1), np.array(labels2)
    
    control_img = np.load('control.npy')

    while True:
        descriptions, features, labels1, labels2 = get_data_from_tip(tip, batch_size)
        # print('data: f min {} max {}, l1 min {} max {}, l2 min {} max {}'.format(features.min(), features.max(), labels1.min(), labels1.max(), labels2.min(), labels2.max()))
        
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
            print(tb)
            open("numpy_warns.log", 'ab').write(str(descriptions).encode('utf-8'))
            open("numpy_warns.log", 'ab').write(str(tb).encode('utf-8'))
            open("numpy_warns.log", 'a').write('------------------------------------')
            continue

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
        