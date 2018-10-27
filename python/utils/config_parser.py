import json
import os
import tensorflow as tf

from utils.graph_utils import ModelMeta, LossMeta, MetricsMeta

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
            'feeders': list, 'losses': list, 'tasks': list}
    for k, t in required_keys.items():
        if k not in config:
            raise ValueError
        if not isinstance(config[k], t):
            raise ValueError

    return config['experiment_name'], config


def get_tasks_info(cfg):
    """Reads the tasks names and additional info (if relevant)
    """
    tasks_info = cfg['tasks']
    tasks_names = []
    for task in tasks_info:
        tasks_names.append(task['name'])
    return tasks_names


def import_model_from_cfg(cfg, sess):
    return ModelMeta(sess, **cfg['model'])


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


def import_losses_from_cfg(cfg, model, ground_truths, task_names):
    """Read loss functions definitions from configuration dictionary
    """
    assert len(model.outputs) == len(task_names)
    assert len(model.outputs) == len(ground_truths)
    losses = [None] * len(model.outputs)

    for loss_config in cfg['losses']:
        task_id = loss_config['attach_to']
        loss_defname = loss_config['definition']
        pred_tensor = model.outputs[task_id]
        true_tensor = ground_truths[task_id]
        name = 'losses/{}'.format(task_names[task_id])
        losses[task_id] = LossMeta(
                loss_defname, pred_tensor, true_tensor, name)

    if model.regularizer is not None:
        for idx in range(len(model.outputs)):
            losses[idx].op += model.regularizer

    return losses


def import_metrics_from_cfg(cfg, model, ground_truths, task_names):
    """Read metrics definitions from configuration dictionary
    """
    assert len(model.outputs) == len(task_names)
    assert len(model.outputs) == len(ground_truths)
    metrics = [[] for i in range(len(model.outputs))]

    for metrics_config in cfg['metrics']:
        task_id = metrics_config['attach_to']
        metrics_defname = metrics_config['definition']
        pred_tensor = model.outputs[task_id]
        true_tensor = ground_truths[task_id]
        name = 'metrics/{}_{}'.format(
                task_names[task_id], len(metrics[task_id]))
        metrics[task_id].append(MetricsMeta(
            metrics_defname, pred_tensor, true_tensor, name))

    return metrics


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


def prepare_feeder_placeholders(outputs, tasks_names):
    with tf.variable_scope('feeders'):
        ground_truths = []
        for idx in range(len(outputs)):
            y = tf.placeholder(
                    shape=outputs[idx].get_shape(), dtype=tf.float32,
                    name='{}_feeder'.format(tasks_names[idx]))
            ground_truths.append(y)
    return ground_truths


class Tasks:
    """
    Just a convenient structure to hold all the necessary details about
    the tasks (names, placeholders, data feeders, losses, metrics, etc.
    in given experiment configuration.
    """
    def __init__(self, cfg, model, batch_size):
        self.names = get_tasks_info(cfg)
        self.gt = prepare_feeder_placeholders(
                model.outputs, self.names)
        self.train_feeders, self.val_feeders = \
            import_feeders_from_cfg(cfg, batch_size)
        self.losses = import_losses_from_cfg(
                cfg, model, self.gt, self.names)
        self.metrics = import_metrics_from_cfg(
                cfg, model, self.gt, self.names)

        assert len(model.outputs) == len(self.names)
        assert len(model.outputs) == len(self.losses)
        assert len(model.outputs) == len(self.metrics)
        assert len(model.outputs) == len(self.train_feeders)
        assert len(model.outputs) == len(self.val_feeders)

        self.n = len(model.outputs)
