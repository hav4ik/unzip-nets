import os
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


def initialize_uninitialized_variables(sess):
    """Initialize uninitialized variables (all variables except those
       that already have weight)
    """
    with tf.variable_scope('var_managers/', reuse=tf.AUTO_REUSE):
        uninitialized_names = sess.run(tf.report_uninitialized_variables())
        for i in range(len(uninitialized_names)):
            uninitialized_names[i] = uninitialized_names[i].decode('utf-8')
            init_op = tf.variables_initializer(
                [v for v in tf.global_variables()
                    if v.name.split(':')[0] in uninitialized_names])
    sess.run(init_op)


class ModelMeta:
    """
    Holds the model's metadata necessary for performing unzipping movements.
    Nothing fancy, just a thin layer between our messy code with TF to keep
    things more or less organized.
    """
    def __init__(self,
                 sess,
                 definition,
                 weights=None,
                 params={}):
        model_name_scope = definition

        with tf.variable_scope(model_name_scope):
            inputs, outputs, update_ops, regularizer = \
                    load_model(definition, weights, params)

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=model_name_scope)

        with tf.variable_scope('var_managers/', reuse=tf.AUTO_REUSE):
            model_saver = tf.train.Saver(self.var_list, max_to_keep=None)

        if weights is not None:
            weights_path = os.path.expanduser(weights)
            if os.path.isdir(weights_path):
                weights_path = tf.train.latest_checkpoint(weights_path)
            print('\nRestoring model from {}'.format(weights_path))
            model_saver.restore(sess, weights_path)

        self.inputs = inputs
        self.outputs = outputs
        self.update_ops = update_ops
        self.regularizer = regularizer
        self.saver = model_saver


class LossMeta:
    """
    Just holds the loss function's metadata that we might need for
    unzipping movements.
    Nothing fancy, just a thin layer between our messy code with TF to keep
    things more or less organized.
    """
    def __init__(self,
                 loss_defname,
                 pred_tensor,
                 true_tensor,
                 name='losses/'):

        if hasattr(tf.keras.losses, loss_defname):
            loss_def = getattr(tf.keras.losses, loss_defname)
        elif hasattr(custom.losses, loss_defname):
            loss_def = getattr(custom.losses, loss_defname)
        else:
            raise ValueError

        with tf.variable_scope(name):
            self.op = tf.reduce_mean(loss_def(true_tensor, pred_tensor))
            self.accumulator, self.resetter, self.average = \
                build_op_accumulator(self.op)


class MetricsMeta:
    """
    Just holds the loss function's any metadata that we might need for
    unzipping movements.
    Nothing fancy, just a thin layer between our messy code with TF to keep
    things more or less organized.
    """
    def __init__(self,
                 metrics_defname,
                 pred_tensor,
                 true_tensor,
                 name='metrics/'):

        if hasattr(tf.keras.metrics, metrics_defname):
            metrics_def = getattr(tf.keras.metrics, metrics_defname)
        elif hasattr(custom.metrics, metrics_defname):
            metrics_def = getattr(custom.metrics, metrics_defname)
        else:
            raise ValueError

        with tf.variable_scope(name):
            self.op = tf.reduce_mean(metrics_def(true_tensor, pred_tensor))
            self.accumulator, self.resetter, self.average = \
                build_op_accumulator(self.op)
