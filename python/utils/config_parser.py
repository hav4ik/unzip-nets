import json
import os
import tensorflow as tf

from utils.graph_utils import build_op_accumulator, get_variables, load_model
import custom.feeders
import custom.losses
import custom.metrics
import custom.schedulers


class dict2obj:
    """Turns dictionary into a class
    """
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return '<dict2obj: %s>' % attrs


def read_config(config):
    """Parse a JSON-formatted configuration, either from file or
    as a string. More on the format of configuration file, please
    visit docs/config_format.md

    Args:
      config: either a JSON string or a path to JSON file

    Returs:
      config_dict: config parsed into python's `dict` structure

    Raises:
      ValueError: if the format requirements of the configuration
                  file is not met
    """
    if os.path.isfile(config):
        config = open(config, 'r').read()
    config = json.loads(config)

    required_keys = {
            'experiment_name': str, 'model': dict, 'metrics': list,
            'feeders': list, 'losses': list}
    for k, t in required_keys.items():
        if k not in config:
            raise ValueError
        if not isinstance(config[k], t):
            raise ValueError

    return config


def import_model_from_cfg(sess, cfg):
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


def import_feeders_from_cfg(cfg, batch_size):
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


def import_losses_from_cfg(cfg, outputs, ground_truths, regularizer):
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
        loss_defs.append(dict2obj(loss_def))

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

            accumulator, resetter, average = build_op_accumulator(
                    losses[loss_def.attach_to])
            accumulators[loss_def.attach_to] = accumulator
            resetters[loss_def.attach_to] = resetter
            averages[loss_def.attach_to] = average

    return losses, accumulators, resetters, averages


def import_metrics_from_cfg(cfg, outputs, ground_truths):
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
        metrics_defs.append(dict2obj(metrics_def))

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

            accumulator, resetter, average = build_op_accumulator(m)
            accumulators[metrics_def.attach_to].append(accumulator)
            averages[metrics_def.attach_to].append(average)
            resetters[metrics_def.attach_to].append(resetter)

    return metrics, accumulators, resetters, averages


def import_optimizers_from_cfg(cfg):
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
