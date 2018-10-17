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


def _read_model_cfg(cfg):
    """Reads model configuration from config dictionary
    """
    model_config = cfg['model']
    inputs, outputs, update_ops, regulizer = load_model(
        model_config['definition'],
        model_config['weights'],
        model_config['params'])
    return inputs, outputs, update_ops, regulizer


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
    for loss_def in loss_defs:
        with tf.name_scope('losses'):
            losses[loss_def.attach_to] = tf.reduce_mean(loss_def.loss(
                ground_truths[loss_def.attach_to],
                outputs[loss_def.attach_to]))
            losses[loss_def.attach_to] *= loss_def.coeff
            if regularizer is not None:
                losses[loss_def.attach_to] += regularizer

    return losses


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
    for metrics_def in metrics_defs:
        with tf.name_scope('metrics'):
            m = tf.reduce_mean(metrics_def.metrics(
                ground_truths[metrics_def.attach_to],
                outputs[metrics_def.attach_to]))
            metrics[metrics_def.attach_to].append(m)

    return metrics


def _read_optimizer_op_cfg(cfg):
    """Loads optimizer ops from keras or custom module
    """
    optimizer_config = cfg['optimizer']
    if hasattr(tf.train, optimizer_config['definition']):
        optimizer_op = getattr(tf.train, optimizer_config['definition'])
        optimizer_params = optimizer_config['params']
    else:
        raise ValueError
    return optimizer_op, optimizer_params


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


def _prepare_dirs(out_dir):
    """Prepares the output directory structure
    """
    tensorboard_dir = os.path.join(out_dir, 'tensorboard')
    checkpoints_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    return tensorboard_dir, checkpoints_dir


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
    tensorboard_dir, checkpoints_dir = _prepare_dirs(out_dir)

    with tf.name_scope(cfg['model']['definition']):
        inputs, outputs, update_ops, regularizer = _read_model_cfg(cfg)
    model_saver = tf.train.Saver(get_variables(cfg['model']['definition']))

    ground_truths = []
    for output in outputs:
        y = tf.placeholder(shape=output.get_shape(), dtype=tf.float32)
        ground_truths.append(y)

    training_feeders, validating_feeders = _read_feeders_cfg(cfg, batch_size)
    optimizer_op, optimizer_params = _read_optimizer_op_cfg(cfg)
    losses = _read_losses_cfg(cfg, outputs, ground_truths, regularizer)
    metrics = _read_metrics_cfg(cfg, outputs, ground_truths)

    train_feeder_idx, train_samples = _concatenate_feeders(training_feeders)
    val_feeder_idx, val_samples = _concatenate_feeders(validating_feeders)

    if mode == 'alternating':
        gradients = []
        targets = []
        for loss in losses:
            with tf.control_dependencies(update_ops):
                optimizer = optimizer_op(**optimizer_params)
                grads_and_vars = optimizer.compute_gradients(loss=loss)
                gradients.append(grads_and_vars)
                targets.append(optimizer.apply_gradients(grads_and_vars))

        _initialize_uninitialized_variables(sess)

        with tf.name_scope('tensorboard'):
            for g, v in gradients[0]:
                if g is None:
                    continue
                tf.summary.histogram("gradients/" + v.name, _l2_filter_norm(g))
                tf.summary.histogram("variables/" + v.name, _l2_filter_norm(v))
            summaries_op = tf.summary.merge_all()

        summaries_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        summary_counter = 0

        prev_val_loss_mean = float('inf')
        for epoch in range(n_epochs):
            print('\nEpoch #{}:'.format(epoch))
            np.random.shuffle(train_feeder_idx)

            trn_losses = np.zeros(shape=(len(outputs)), dtype=np.float64)
            trn_metrics = [
                np.zeros((len(metrics[i]))) for i in range(len(outputs))]
            train_idxs = tqdm(train_feeder_idx[:], desc='trn')

            for idx in train_idxs:
                x, y = next(training_feeders[idx])
                ops_to_run = {'loss': losses[idx],
                              'metrics': metrics[idx],
                              'optimizer': targets[idx]}
                summary_counter += 1
                if summary_counter % 100 == 0 and idx == 0:
                    ops_to_run['summaries'] = summaries_op

                feed_dict = {inputs[0]: x,
                             ground_truths[idx]: y,
                             K.learning_phase(): 1}
                out = sess.run(ops_to_run, feed_dict=feed_dict)

                if 'summaries' in out:
                    summaries_writer.add_summary(
                            out['summaries'], summary_counter)

                trn_losses[idx] += out['loss']
                trn_metrics[idx] += out['metrics']

            val_losses = np.zeros(shape=(len(outputs)), dtype=np.float64)
            val_metrics = [
                    np.zeros((len(metrics[i]))) for i in range(len(outputs))]
            val_idxs = tqdm(val_feeder_idx[:], desc='val')
            for idx in val_idxs:
                x, y = next(validating_feeders[idx])

                ops_list = [losses[idx]] + metrics[idx]
                feed_dict = {inputs[0]: x,
                             ground_truths[idx]: y,
                             K.learning_phase(): 0}
                out = sess.run(ops_list, feed_dict=feed_dict)

                val_losses[idx] += out[0]
                val_metrics[idx] += out[1:]

            trn_losses /= train_samples
            trn_metrics /= train_samples
            val_losses /= val_samples
            val_metrics /= val_samples

            for i in range(len(outputs)):
                print('task #{}:'.format(i))
                print('  -(train): loss={:3.6f}, metrics=[{}]'.format(
                    trn_losses[i],
                    ', '.join(['{:.6f}'.format(x) for x in trn_metrics[i]])))
                print('  -(val)  : loss={:3.6f}, metrics=[{}]'.format(
                    val_losses[i],
                    ', '.join(['{:.6f}'.format(x) for x in val_metrics[i]])))

            val_losses_mean = val_losses.mean()
            if val_losses_mean < prev_val_loss_mean:
                print('The validation loss mean has improved from {:.6f} to '
                      '{:.6f}. Saving checkpoints.'.format(prev_val_loss_mean,
                                                           val_losses_mean))
                prev_val_loss_mean = val_losses_mean
                model_saver.save(sess, os.path.join(
                    checkpoints_dir, 'mtl'), global_step=epoch)

    elif mode == 'joint':
        pass

    else:
        raise ValueError


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
