import os
import numpy as numpy
import tensorflow as tf

import custom.models


def load_model(model, weights=None, params={}):
    """loads a model either from module or from a checkpoint
    """
    if hasattr(custom.models, model):
        model_definition = getattr(custom.models, model)
        loaded_model = model_definition(**params)
    else:
        pass

    return loaded_model


def get_variables(name_scope):
    """Gets all variables in the namescope
    """
    variables = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=name_scope):
        variables.append(var)
    return variables


def build_op_accumulator(op):
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


def initialize_uninitialized_variables(sess):
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
