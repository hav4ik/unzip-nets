import os
from argparse import ArgumentParser
import numpy as np
from collections import namedtuple
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
    inputs, outputs, update_ops = load_model(model_config['definition'],
            model_config['weights'], model_config['params'])
    return inputs, outputs, update_ops


def _read_feeders_cfg(cfg, batch_size):
    """Reads feeder configurations from config dictionary
    """
    training_feeders = []
    validating_feeders = []
    for feeder_config in cfg['feeders']:
        training_cfg = dict(feeder_config['params']['common'],
                **feeder_config['params']['training'])
        validating_cfg = dict(feeder_config['params']['common'],
                **feeder_config['params']['validating'])
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
        metrics_def = {'metrics': metrics,
                'attach_to': metrics_config['attach_to']}
        metrics_defs.append(config_parser.dict2obj(metrics_def))
    return metrics_defs


def _read_optimizer_op_cfg(cfg):
    optimizer_config = cfg['optimizer']
    if hasattr(tf.train, optimizer_config['definition']):
        optimizer_op = getattr(tf.train, optimizer_config['definition'])
        optimizer_params = optimizer_config['params']
    else:
        raise ValueError
    return optimizer_op, optimizer_params


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
    assert len(outputs) == len(training_feeders)
    assert len(outputs) == len(validating_feeders)
    assert len(outputs) == len(loss_defs)

    ground_truths = []
    for output in outputs:
        y = tf.placeholder(shape=output.get_shape(), dtype=tf.float32)
        ground_truths.append(y)

    losses = [None] * len(outputs)
    for loss_def in loss_defs:
        losses[loss_def.attach_to] = loss_def.coeff * tf.reduce_mean(loss_def.loss(
            ground_truths[loss_def.attach_to], outputs[loss_def.attach_to]))

    metrics = [[]] * len(outputs)
    for metrics_def in metrics_defs:
        metrics[metrics_def.attach_to].append(metrics_def.metrics(
            ground_truths[metrics_def.attach_to], outputs[metrics_def.attach_to]))

    if mode == 'alternating':
        targets = []
        for loss in losses:
            targets.append(optimizer_op(**optimizer_params).minimize(loss))

        writer = tf.summary.FileWriter(out_dir, sess.graph)
        writer.close()

        n_samples = [f.n // batch_size for f in training_feeders]
        feeder_idx = np.zeros(shape=(sum(n_samples)), dtype=np.uint32)
        accumulated = 0
        for i in range(len(n_samples)):
            feeder_idx[accumulated:accumulated+n_samples[i]] = i

        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(n_epochs):
            np.random.shuffle(feeder_idx)

            idxs = tqdm(feeder_idx, ascii=True, desc='epoch #{}'.format(epoch))
            for idx in idxs:
                x, y = next(training_feeders[idx])
                out = sess.run([losses[idx]] + metrics[idx] + [targets[idx]],
                        feed_dict={inputs[0]: x, ground_truths[idx]: y})



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='A JSON file or a string in JSON format')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-n', '--epochs', type=int, default=1)
    parser.add_argument('-m', '--mode', type=str, default='alternating')
    parser.add_argument('-o', '--out_dir', type=str, default='out/')
    args = parser.parse_args()

    train_multitask(args.config, args.batch_size, args.epochs, args.out_dir, args.mode)
