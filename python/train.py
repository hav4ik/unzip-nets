import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import config_parser
from utils.model_utils import load_model
import custom.feeders
import custom.losses


def _read_model_cfg(cfg):
    """Reads model configuration from config dictionary
    """
    model_config = cfg['model']
    inputs, outputs, update_ops = load_model(
            model_config['definition'],
            model_config['weights'],
            model_config['params'])
    return inputs, outputs, update_ops


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


def _read_loss_defs_cfg(cfg):
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
        loss_def = {
                'loss': loss,
                'attach_to': loss_config['attach_to'],
                'coeff': loss_config['coeff']}
        loss_defs.append(config_parser.dict2obj(loss_def))
    return loss_defs


def _read_metrics_defs_cfg(cfg):
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
        metrics_def = {
                'metrics': metrics,
                'attach_to': metrics_config['attach_to']}
        metrics_defs.append(config_parser.dict2obj(metrics_def))
    return metrics_defs


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


def _initialize_variables(sess):
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
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    cfg = config_parser.read_config(config)

    inputs, outputs, update_ops = _read_model_cfg(cfg)
    training_feeders, validating_feeders = _read_feeders_cfg(cfg, batch_size)
    loss_defs = _read_loss_defs_cfg(cfg)
    metrics_defs = _read_metrics_defs_cfg(cfg)
    optimizer_op, optimizer_params = _read_optimizer_op_cfg(cfg)

    ground_truths = []
    for output in outputs:
        y = tf.placeholder(shape=output.get_shape(), dtype=tf.float32)
        ground_truths.append(y)

    losses = [None] * len(outputs)
    for loss_def in loss_defs:
        with tf.name_scope('losses'):
            losses[loss_def.attach_to] = tf.reduce_mean(loss_def.loss(
                    ground_truths[loss_def.attach_to],
                    outputs[loss_def.attach_to]))
            losses[loss_def.attach_to] *= loss_def.coeff

    metrics = [[] for i in range(len(outputs))]
    for metrics_def in metrics_defs:
        with tf.name_scope('metrics'):
            m = tf.reduce_mean(metrics_def.metrics(
                    ground_truths[metrics_def.attach_to],
                    outputs[metrics_def.attach_to]))
            metrics[metrics_def.attach_to].append(m)

    train_feeder_idx, train_samples = _concatenate_feeders(training_feeders)
    val_feeder_idx, val_samples = _concatenate_feeders(validating_feeders)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if mode == 'alternating':
        targets = []
        for loss in losses:
            targets.append(optimizer_op(**optimizer_params).minimize(loss))
        _initialize_variables(sess)

        writer = tf.summary.FileWriter(out_dir, sess.graph)
        writer.close()

        for epoch in range(n_epochs):
            print('Epoch #{}:'.format(epoch))
            np.random.shuffle(train_feeder_idx)

            train_idxs = tqdm(train_feeder_idx[:], desc='trn')
            for idx in train_idxs:
                x, y = next(training_feeders[idx])
                with tf.control_dependencies(update_ops):
                    ops_list = [losses[idx]] + metrics[idx] + [targets[idx]]
                    feed_dict = {
                            inputs[0]: x,
                            ground_truths[idx]: y,
                            K.learning_phase(): 1}
                    out = sess.run(ops_list, feed_dict=feed_dict)

            val_losses = [0] * len(outputs)
            val_metrics = [np.zeros((len(metrics[i])))
                    for i in range(len(outputs))]
            val_idxs = tqdm(val_feeder_idx[:], desc='val')
            for idx in val_idxs:
                x, y = next(validating_feeders[idx])
                ops_list = [losses[idx]] + metrics[idx]
                feed_dict = {
                        inputs[0]: x,
                        ground_truths[idx]: y,
                        K.learning_phase(): 0}
                out = sess.run(ops_list, feed_dict=feed_dict)
                val_losses[idx] += out[0]
                val_metrics[idx] += out[1:]

            print('losses:', ', '.join([
                    '{:.6f}'.format(val_losses[i] / val_samples[i])
                    for i in range(len(outputs))]))
            print('metrics:')
            for i in range(len(outputs)):
                print(val_metrics[i] / val_samples[i])

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
