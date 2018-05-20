import warnings as w
w.simplefilter(action = 'ignore', category = FutureWarning)

from six.moves import xrange
import better_exceptions
import datetime
from functools import partial
import os

import yaml
import matplotlib.pyplot as plt
import json
import shutil
import traceback


def _arch_type_a(num_classes):
    def _ru(t, conv3_1, bn_1, conv3_2, bn_2):
        _t = conv3_1(t)
        _t = bn_1(_t)
        _t = tf.nn.relu(_t)
        _t = conv3_2(_t)
        _t = bn_2(_t)
        return t + _t

    def _frru(y_z, conv3_1, bn_1, conv3_2, bn_2, conv1, scale):
        # tf.nn.max_pool(, ksize, strides, padding, data_format='NHWC', name=None)
        y, z = y_z

        _t = tf.concat([y,
                        tf.nn.max_pool(z, [1, scale, scale, 1], [1, scale, scale, 1], 'SAME', 'NHWC')], axis=3)
        _t = conv3_1(_t)
        _t = bn_1(_t)
        _t = tf.nn.relu(_t)
        _t = conv3_2(_t)
        _t = bn_2(_t)
        y_prime = tf.nn.relu(_t)

        _t = conv1(y_prime)
        _t = tf.image.resize_nearest_neighbor(_t, tf.shape(y_prime)[1:3]*scale)
        z_prime = _t + z

        return y_prime, z_prime

    def _divide_stream(t, conv1):
        z = conv1(t)
        return t, z

    def _concat_stream(y_z, conv1):
        y, z = y_z
        t = tf.concat([tf.image.resize_bilinear(
            y, tf.shape(y)[1:3]*2), z], axis=3)
        return conv1(t)

    from functools import partial
    from commons.ops import Conv2d, BatchNorm
    import tensorflow as tf
    
    # The First Conv
    spec = [
        Conv2d('conv2d_1', 1, 48, 5, 5, 1, 1, data_format='NHWC'),
        BatchNorm('conv2d_1_bn', 48, axis=3),
        lambda t, **kwargs: tf.nn.relu(t)]
    # RU Layers
    for i in range(3):
        spec.append(
            partial(_ru,
                    conv3_1=Conv2d('ru48_%d_1' % i, 48, 48, 3,
                                   3, 1, 1, data_format='NHWC'),
                    bn_1=BatchNorm('ru48_%d_1_bn' % i, 48, axis=3),
                    conv3_2=Conv2d('ru48_%d_2' % i, 48, 48, 3,
                                   3, 1, 1, data_format='NHWC'),
                    bn_2=BatchNorm('ru48_%d_2_bn' % i, 48, axis=3))
        )
    # Split Streams
    spec.append(
        partial(_divide_stream,
                conv1=Conv2d('conv32', 48, 32, 1, 1, 1, 1, data_format='NHWC'))
    )
    # FFRU Layers (Encoding)
    prev_ch = 48
    for it, ch, scale in [(3, 96, 2), (4, 192, 4), (2, 384, 8), (2, 384, 16)]:
        spec.append(
            # maxpooling y only.
            lambda y_z: (tf.nn.max_pool(y_z[0], [1, 2, 2, 1], [
                         1, 2, 2, 1], 'SAME', 'NHWC'), y_z[1])
        )
        for i in range(it):
            spec.append(
                partial(_frru,
                        conv3_1=Conv2d('encode_frru%d_%d_%d_1' % (
                            ch, scale, i), prev_ch+32, ch, 3, 3, 1, 1, data_format='NHWC'),
                        bn_1=BatchNorm('encode_frru%d_%d_%d_1_bn' %
                                       (ch, scale, i), ch, axis=3),
                        conv3_2=Conv2d('encode_frru%d_%d_%d_2' % (
                            ch, scale, i), ch, ch, 3, 3, 1, 1, data_format='NHWC'),
                        bn_2=BatchNorm('encode_frru%d_%d_%d_2_bn' %
                                       (ch, scale, i), ch, axis=3),
                        conv1=Conv2d('encode_frru%d_%d_%d_3' % (
                            ch, scale, i), ch, 32, 1, 1, 1, 1, data_format='NHWC'),
                        scale=scale)
            )
            prev_ch = ch
    # FRRU Layers (Decoding)
    for it, ch, scale in [(2, 192, 8), (2, 192, 4), (2, 96, 2)]:
        spec.append(
            lambda y_z: (tf.image.resize_bilinear(
                y_z[0], tf.shape(y_z[0])[1:3]*2), y_z[1])
        )
        for i in range(it):
            spec.append(
                partial(_frru,
                        conv3_1=Conv2d('decode_frru%d_%d_%d_1' % (
                            ch, scale, i), prev_ch+32, ch, 3, 3, 1, 1, data_format='NHWC'),
                        bn_1=BatchNorm('decode_frru%d_%d_%d_1_bn' %
                                       (ch, scale, i), ch, axis=3),
                        conv3_2=Conv2d('decode_frru%d_%d_%d_2' % (
                            ch, scale, i), ch, ch, 3, 3, 1, 1, data_format='NHWC'),
                        bn_2=BatchNorm('decode_frru%d_%d_%d_2_bn' %
                                       (ch, scale, i), ch, axis=3),
                        conv1=Conv2d('decode_frru%d_%d_%d_3' % (
                            ch, scale, i), ch, 32, 1, 1, 1, 1, data_format='NHWC'),
                        scale=scale)
            )
            prev_ch = ch
    # Concat Streams
    spec.append(
        partial(_concat_stream,
                conv1=Conv2d('conv48', prev_ch+32, 48, 1, 1, 1, 1, data_format='NHWC')))
    # RU Layers
    for i in range(3, 6):
        spec.append(
            partial(_ru,
                    conv3_1=Conv2d('ru48_%d_1' % i, 48, 48, 3,
                                   3, 1, 1, data_format='NHWC'),
                    bn_1=BatchNorm('ru48_%d_1_bn' % i, 48, axis=3),
                    conv3_2=Conv2d('ru48_%d_2' % i, 48, 48, 3,
                                   3, 1, 1, data_format='NHWC'),
                    bn_2=BatchNorm('ru48_%d_2_bn' % i, 48, axis=3))
        )
    # Final Classification Layer
    spec.append(
        Conv2d('conv_c', 48, num_classes, 1, 1, 1, 1, data_format='NHWC'))

    return spec


class FRRN():
    def __init__(self, K, im, gt, arch_fn, param_scope):
        import tensorflow as tf
        # with tf.variable_scope(param_scope):

        # with tf.variable_scope('forward') as forward_scope:
        net_spec = arch_fn()
        _t = im
        for block in net_spec:
            print(_t)
            _t = block(_t)
        self.logits = _t
        self.preds = tf.argmax(self.logits, axis=3)

        # Loss
        self.naive_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=gt)
        # TODO: ignore pixels labed as void? is it requried?
        # mask = tf.logical_not(tf.equal(gt,0))
        # naive_loss = naive_loss * mask
        shape = [tf.shape(im)[0], tf.shape(im)[1]*tf.shape(im)[2]]
        boot_loss, _ = tf.nn.top_k(tf.reshape(self.naive_loss, shape), k=K, sorted=False)
        self.loss = tf.reduce_mean(tf.reduce_sum(boot_loss, axis=1))

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

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    from pipeline import Pipeline
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap
    from commons.ops import *
    # with tf.variable_scope('params') as params:
    #     pass

    IM_HEIGHT = 224
    IM_WIDTH = 224
    IM_CHANNEL = 1
    batch_size = 24
    iteration = 0
    min_loss = 1e10
    checkpoint_base = 'D:\\checkpoint_frrn'
    startup_file_name = 'startup.frrn.json'
    models_to_keep = 3
    models_history = []
    checkpoint_steps = 250
    model_version = 1
    learning_rate = 1e-5
    K = IM_WIDTH * 64
    num_classes = 2
    time_str = datetime.datetime.now().strftime('%m-%d--%H%M%S')
    log_string = 'logs/frnn-{}/{}-lr-{:.8f}'.format(model_version,
                                                    time_str,
                                                    learning_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)


    im = tf.placeholder(tf.float32, [batch_size, IM_HEIGHT, IM_WIDTH, IM_CHANNEL])
    # 19 + unlabeled area(plus ignored labels)
    gt = tf.placeholder(tf.int32, [batch_size, IM_HEIGHT, IM_WIDTH])
    global_step = tf.Variable(0, trainable=False)
    summary = []
    devices = ['/gpu:0', '/gpu:1']
    x_split = tf.split(im, len(devices))
    y_split = tf.split(gt, len(devices))

    writer = tf.summary.FileWriter(log_string)

    all_logits = []
    all_loss = []
    all_preds = []
    eval_logits = []
    tower_grads = []

    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.variable_scope(tf.get_variable_scope()):
        for i, (device, x, y) in enumerate(zip(devices, x_split, y_split)):
            with tf.device(device):
                with tf.name_scope('tower_%d' % i) as scope:
                    print(i, scope)
                    net = FRRN(K, x, y, partial(_arch_type_a, num_classes), scope)
                    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    # updates_op = tf.group(*update_ops)
                    # with tf.control_dependencies([updates_op]):
                    #     losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                    #     total_loss = tf.add_n(losses, name='total_loss')
                    grads = optimizer.compute_gradients(net.loss)
                    # grads = optimizer.compute_gradients(total_loss)
                    
                    tf.get_variable_scope().reuse_variables()
                    
                    eval_logits.append(net.logits)
                    all_loss.append(net.loss)
                    all_preds.append(net.preds)
                    tower_grads.append(grads)
                    # all_predictions['classes'].append(tf.argmax(net.logits, axis=1))
    
    vars_ = tf.trainable_variables()    
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    # print('towers:', tower_grads)
    # import pdb; pdb.set_trace()
    loss = tf.reduce_mean(all_loss)
    preds = tf.concat(all_preds, axis=0)
    logits = tf.concat(eval_logits, axis=0)
    grads_ = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(grads_, global_step=global_step)

    summary = []
    for i, (loss_, preds_, grad) in enumerate(zip(all_loss, all_preds, tower_grads)):
        with tf.name_scope('tower_%d' % i):
            summary.append(tf.summary.histogram('preds', preds_))
            summary.append(tf.summary.scalar('loss', loss_))
            grad_norm = tf.norm(grad[0][0])
            summary.append(tf.summary.scalar('gradient', grad_norm))

    with tf.name_scope('all'):
        summary.append(tf.summary.scalar('loss', loss))
        grad_norm = tf.norm(grads_[0][0])
        summary.append(tf.summary.scalar('gradient', grad_norm))
    summary = tf.summary.merge(summary)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess.run(init_op)
    
    try:
        print('[+] loading startup.json')
        startup = json.load(open('startup.frrn.json', 'r'))
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
        
        min_loss = state['min_loss']
        checkpoint_path = state['checkpoint_path']
        sess.run(tf.assign(global_step, state['iteration']))
    except:
        print('[!] no models to checkpoint from..')
        # raise
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    sess.graph.finalize()

    with open('configs/win_client.frrn.yaml') as f:
        config = yaml.safe_load(f)
    amp = Pipeline(config['pipeline'])
    amp.run()
    amp.keepalive(block=False)
    tip = amp._get_tip_queue()

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

    seq = iaa.Sequential(iaa.OneOf(augmenters),)


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
    
    while True:
        descriptions, features, labels = get_data_from_tip(tip, batch_size)

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
            continue

        to_run = [logits, preds, train_op, summary, loss, global_step]
        # import pdb; pdb.set_trace()
        _t, _preds, _, _summary, _loss, iteration = sess.run(to_run,
                                                             feed_dict={
                                                                 im: features.reshape((batch_size, IM_HEIGHT, IM_WIDTH, 1)),
                                                                 gt: labels.reshape((batch_size, IM_HEIGHT, IM_WIDTH))
                                                             })

        # import pdb; pdb.set_trace()

        #    feed_dict={im: np.random.rand(batch_size, IM_HEIGHT, IM_WIDTH, 1),
        #               gt: np.random.rand(batch_size, IM_HEIGHT, IM_WIDTH)}))
        min_loss = min(min_loss, _loss)
        print('[+] Iteration {:10}, loss {:<17}, min_loss {:<17}'.format(iteration, _loss, min_loss))
        if iteration % 10 == 0:
            writer.add_summary(_summary, iteration)

        if iteration % checkpoint_steps == 0:
            # print('\t\tNew Best Loss!')
            # best_loss = train_loss
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            
            checkpoint_dir = os.path.join(checkpoint_base, timestamp)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            print('\t\tSaving model to:' + checkpoint_path)
            saver.save(sess, checkpoint_path, global_step=global_step)
            state = {
                'iteration': int(iteration),
                # 'val_acc': float(val_acc),
                'min_loss': int(min_loss),
                'checkpoint_path': checkpoint_path,
                'log_string': log_string,
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

        
        # iteration += 1
