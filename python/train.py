import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import config_parser
from utils.model_utils import load_model, get_variables
import custom.feeders
import custom.losses
import custom.schedulers


def _read_model_cfg(sess, cfg):
    """Reads model configuration from config dictionary
    """
    model_config = cfg['model']
    with tf.name_scope(model_config['definition']):
        inputs, outputs, update_ops, regulizer = load_model(
                model_config['definition'],
                model_config['weights'],
                model_config['params'])

    model_saver = tf.train.Saver(
            get_variables(model_config['definition']),
            max_to_keep=None)

    if model_config['weights'] is not None:
        weights_path = os.path.expanduser(model_config['weights'])
        if os.path.isdir(weights_path):
            weights_path = tf.train.latest_checkpoint(weights_path)
        print('\nRestoring model from {}'.format(weights_path))
        model_saver.restore(sess, weights_path)

    return inputs, outputs, update_ops, regulizer, model_saver


def _read_feeders_cfg(cfg, batch_size):
    """Reads feeder configurations from config dictionary
    """
    training_feeders = []
    validating_feeders = []
    for feeder_config in cfg['feeders']:
        common = feeder_config['params']['common']
        training_cfg = feeder_config['params']['training']
        validating_cfg = feeder_config['params']['validating']
        training_cfg.update(common)
        validating_cfg.update(common)
        feeder = getattr(custom.feeders, feeder_config['definition'])

        training_feeders.append(
            feeder(batch_size, 'training', **training_cfg))
        validating_feeders.append(
            feeder(batch_size, 'validating', **validating_cfg))

    return training_feeders, validating_feeders


def _build_accumulator(op):
    """For updating and averaging gradients, metrics, losses, etc.
       over the whole dataset.
    """
    total = tf.Variable(0., name='total')
    counter = tf.Variable(0., name='counter')
    accumulator_op = tf.group(
            tf.assign_add(total, op),
            tf.assign_add(counter, 1))
    resetter_op = tf.group(
            tf.assign(total, 0),
            tf.assign(counter, 0))
    average = total / counter + 1e-8
    return accumulator_op, resetter_op, average


def _read_losses_cfg(cfg, outputs, ground_truths, regularizer):
    """Read loss functions definitions from configuration dictionary
    """
    loss_defs = []
    for loss_config in cfg['losses']:
        if hasattr(tf.keras.losses, loss_config['definition']):
            loss = getattr(tf.keras.losses, loss_config['definition'])
        elif hasattr(custom.losses, loss_config['definition']):
            loss = getattr(custom.losses, loss_config['definition'])
        else:
            raise ValueError
        loss_def = {'loss': loss,
                    'attach_to': loss_config['attach_to'],
                    'coeff': loss_config['coeff']}
        loss_defs.append(config_parser.dict2obj(loss_def))

    losses = [None] * len(outputs)
    accumulators = [None] * len(outputs)
    averages = [None] * len(outputs)
    resetters = [None] * len(outputs)
    with tf.name_scope('losses'):
        for loss_def in loss_defs:
            losses[loss_def.attach_to] = tf.reduce_mean(loss_def.loss(
                ground_truths[loss_def.attach_to],
                outputs[loss_def.attach_to]))
            losses[loss_def.attach_to] *= loss_def.coeff
            if regularizer is not None:
                losses[loss_def.attach_to] += regularizer

            accumulator, resetter, average = _build_accumulator(
                    losses[loss_def.attach_to])
            accumulators[loss_def.attach_to] = accumulator
            resetters[loss_def.attach_to] = resetter
            averages[loss_def.attach_to] = average

    return losses, accumulators, resetters, averages


def _read_metrics_cfg(cfg, outputs, ground_truths):
    """Read metrics definitions from configuration dictionary
    """
    metrics_defs = []
    for metrics_config in cfg['metrics']:
        if hasattr(tf.keras.metrics, metrics_config['definition']):
            metrics = getattr(tf.keras.metrics, metrics_config['definition'])
        elif hasattr(custom.metrics, metrics_config['definition']):
            metrics = getattr(custom.metrics, metrics_config['definition'])
        else:
            raise ValueError
        metrics_def = {'metrics': metrics,
                       'attach_to': metrics_config['attach_to']}
        metrics_defs.append(config_parser.dict2obj(metrics_def))

    metrics = [[] for i in range(len(outputs))]
    accumulators = [[] for i in range(len(outputs))]
    averages = [[] for i in range(len(outputs))]
    resetters = [[] for i in range(len(outputs))]
    with tf.name_scope('metrics'):
        for metrics_def in metrics_defs:
            m = tf.reduce_mean(metrics_def.metrics(
                ground_truths[metrics_def.attach_to],
                outputs[metrics_def.attach_to]))
            metrics[metrics_def.attach_to].append(m)

            accumulator, resetter, average = _build_accumulator(m)
            accumulators[metrics_def.attach_to].append(accumulator)
            averages[metrics_def.attach_to].append(average)
            resetters[metrics_def.attach_to].append(resetter)

    return metrics, accumulators, resetters, averages


def _read_optimizer_op_cfg(cfg):
    """Loads optimizer ops from keras or custom module
    """
    optimizer_config = cfg['optimizer']
    if hasattr(tf.train, optimizer_config['definition']):
        optimizer_op = getattr(tf.train, optimizer_config['definition'])
        optimizer_params = optimizer_config['params']

        lr_placeholder = None
        if 'learning_rate' in optimizer_params:
            if isinstance(optimizer_params['learning_rate'], str):
                lr_placeholder = tf.placeholder(
                    shape=(), dtype=tf.float32, name='learning_rate')
                lr_scheduler = getattr(custom.schedulers,
                                       optimizer_params['learning_rate'])
                optimizer_params['learning_rate'] = lr_placeholder
        else:
            lr_scheduler = None
    else:
        raise ValueError
    return optimizer_op, optimizer_params, lr_placeholder, lr_scheduler


def _concatenate_feeders(feeders):
    """Make a index array to make a data feeder stream
    """
    n_samples = np.array([f.n // f.batch_size for f in feeders])
    feeder_idx = np.zeros(shape=(sum(n_samples)), dtype=np.uint32)
    accumulated = 0
    for i in range(len(n_samples)):
        feeder_idx[accumulated:accumulated+n_samples[i]] = i
        accumulated += n_samples[i]
    return feeder_idx, n_samples


def _initialize_uninitialized_variables(sess):
    """Initialize uninitialized variables (all variables except those
       that already have weight)
    """
    uninitialized_names = sess.run(tf.report_uninitialized_variables())
    for i in range(len(uninitialized_names)):
        uninitialized_names[i] = uninitialized_names[i].decode('utf-8')
    init_op = tf.variables_initializer(
        [v for v in tf.global_variables()
            if v.name.split(':')[0] in uninitialized_names])
    sess.run(init_op)


def _prepare_dirs(cfg, out_dir):
    """Prepares the output directory structure; the directory is
       uniquely numbered.
    """
    tensorboard_dir = os.path.join(
            os.path.expanduser(out_dir), 'tensorboard', cfg['experiment_name'])
    checkpoints_dir = os.path.join(
            os.path.expanduser(out_dir), 'checkpoints', cfg['experiment_name'])

    for i in range(1001):
        num_tbrd_dir = tensorboard_dir + '-{:03d}'.format(i)
        num_ckpt_dir = checkpoints_dir + '-{:03d}'.format(i)
        if not os.path.isdir(num_tbrd_dir) and not os.path.isdir(num_ckpt_dir):
            break
    if i == 1000:
        raise NameError('There are 999 experiments with the same name already.'
                        ' Please use another name for your experiments.')

    os.makedirs(num_tbrd_dir)
    os.makedirs(num_ckpt_dir)

    return num_tbrd_dir, num_ckpt_dir


def _l2_filter_norm(t):
    return tf.sqrt(tf.reduce_sum(
        tf.pow(t, 2),
        np.arange(len(t.get_shape()) - 1, dtype=np.int32)))


def _prepare_environment():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    return sess


def train_multitask(config,
                    batch_size,
                    n_epochs,
                    out_dir,
                    mode='',
                    verbose=2):
    """Trains an arbitrary model

    Args:
      config:      either a JSON file or a string in JSON format
      batch_size:  batch size for both training and testing
      n_epochs:    number of epochs to train
      mode:        one of the following: 'alternating', 'joint', more in docs
      verbose:     from 0 to 2
    """
    sess = _prepare_environment()

    cfg = config_parser.read_config(config)
    tensorboard_dir, checkpoints_dir = _prepare_dirs(cfg, out_dir)

    (inputs, outputs, update_ops,
        regularizer, model_saver) = _read_model_cfg(sess, cfg)

    ground_truths = []
    for output in outputs:
        y = tf.placeholder(shape=output.get_shape(), dtype=tf.float32)
        ground_truths.append(y)

    training_feeders, validating_feeders = _read_feeders_cfg(cfg, batch_size)
    optimizer_op, optimizer_params, lr_p, lr_s = _read_optimizer_op_cfg(cfg)

    losses, loss_accum, loss_reset, loss_avg = _read_losses_cfg(
            cfg, outputs, ground_truths, regularizer)
    metrics, metrics_accum, metrics_reset, metrics_avg = _read_metrics_cfg(
            cfg, outputs, ground_truths)
    loss_and_metric_reset_ops = (loss_reset, metrics_reset)
    loss_and_metric_avg_ops = (loss_avg, metrics_avg)

    train_feeder_idx, train_samples = _concatenate_feeders(training_feeders)
    val_feeder_idx, val_samples = _concatenate_feeders(validating_feeders)

    if mode == 'alternating':
        print('\nStarted training in mode "Alternating"')
        print('checkpoints: {}'.format(checkpoints_dir))
        print('logs: {}'.format(tensorboard_dir))

        gradients = []
        targets = []
        for loss in losses:
            with tf.control_dependencies(update_ops):
                optimizer = optimizer_op(**optimizer_params)
                grads_and_vars = optimizer.compute_gradients(loss=loss)
                gradients.append(grads_and_vars)
                targets.append(optimizer.apply_gradients(grads_and_vars))

        _initialize_uninitialized_variables(sess)

        with tf.name_scope('summaries'):
            for g, v in gradients[0]:
                if g is None:
                    continue
                tf.summary.histogram("variables/" + v.name, _l2_filter_norm(v))

            summaries_op = tf.summary.merge_all()
            summaries_writer = tf.summary.FileWriter(tensorboard_dir,
                                                     sess.graph)

        for epoch in range(n_epochs):
            if lr_s is not None:
                current_lr = lr_s(epoch)
            else:
                current_lr = 'Auto'
            print('\nEpoch #{}: (lr={})'.format(epoch, current_lr))

            np.random.shuffle(train_feeder_idx)
            train_idxs = tqdm(train_feeder_idx[:], desc='trn')
            sess.run(loss_and_metric_reset_ops)
            for idx in train_idxs:
                x, y = next(training_feeders[idx])
                ops_to_run = {'loss_accumulate': loss_accum[idx],
                              'metrics_accumulate': metrics_accum[idx],
                              'optimizer': targets[idx]}
                feed_dict = {inputs[0]: x,
                             ground_truths[idx]: y,
                             K.learning_phase(): 1}
                if lr_p is not None and lr_s is not None:
                    feed_dict.update({lr_p: current_lr})
                out = sess.run(ops_to_run, feed_dict=feed_dict)
            trn_losses, trn_metrics = sess.run(loss_and_metric_avg_ops)

            val_idxs = tqdm(val_feeder_idx[:], desc='val')
            sess.run(loss_and_metric_reset_ops)
            for idx in val_idxs:
                x, y = next(validating_feeders[idx])
                ops_to_run = {'loss_accumulate': loss_accum[idx],
                              'metrics_accumulate': metrics_accum[idx]}
                feed_dict = {inputs[0]: x,
                             ground_truths[idx]: y,
                             K.learning_phase(): 0}
                out = sess.run(ops_to_run, feed_dict=feed_dict)
            val_losses, val_metrics = sess.run(loss_and_metric_avg_ops)

            summaries_writer.add_summary(sess.run(summaries_op), epoch)

            for i in range(len(outputs)):
                print('task #{}:'.format(i))
                print('  -(train): loss={:3.6f}, metrics=[{}]'.format(
                    trn_losses[i],
                    ', '.join(['{:.6f}'.format(x) for x in trn_metrics[i]])))
                print('  -(val)  : loss={:3.6f}, metrics=[{}]'.format(
                    val_losses[i],
                    ', '.join(['{:.6f}'.format(x) for x in val_metrics[i]])))

            model_saver.save(
                    sess, os.path.join(checkpoints_dir, 'mtl'),
                    write_meta_graph=False, global_step=epoch)

    else:
        raise NameError('Training mode "{}" not supported yet.'.format(mode))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            'config', type=str, help='A JSON file or a string in JSON format')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-n', '--epochs', type=int, default=10)
    parser.add_argument('-m', '--mode', type=str, default='alternating')
    parser.add_argument('-o', '--out_dir', type=str, default='out/')
    args = parser.parse_args()

    train_multitask(args.config,
                    args.batch_size,
                    args.epochs,
                    args.out_dir,
                    args.mode)
