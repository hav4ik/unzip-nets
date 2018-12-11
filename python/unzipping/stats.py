import os
import tensorflow as tf
import numpy as np
import numpy.linalg as lg
from tensorflow.keras import backend as K

from tqdm import tqdm
from utils import graph_utils


def _prepare_dirs(experiment_name, out_dir):
    """Prepares the output directory structure; the directory is
       uniquely numbered.
    """
    stats_dir = os.path.join(
            os.path.expanduser(out_dir), 'stats', experiment_name)

    for i in range(1001):
        num_stats_dir = stats_dir + '-{:03d}'.format(i)
        if not os.path.isdir(num_stats_dir):
            break
    if i == 1000:
        raise NameError('There are 999 experiments with the same name already.'
                        ' Please use another name for your experiments.')

    os.makedirs(num_stats_dir)
    return num_stats_dir


def integral_stats(sess,
                   model,
                   layer_name,
                   tasks,
                   out_dir,
                   steps_per_epoch=None):
    """
    Given a multi-task network `model` and the bottleneck layer `layer_name`,
    calculates the path length of single tasks after an epoch of independent
    training, the distance between tasks after an epoch of training, and the
    angle between paths of individual tasks after an epoch.

    Args:
      sess:             TensorFlow session
      model:            a `graph_utils.ModelMeta` class instance
      layer_name:       the bottleneck layer to monitor
      tasks:            a `config_parser.Tasks` class instance
      out_dir:          Deprecated parameter
      steps_per_epoch:  Number of steps per one epoch. If None, dataset size
                        will be used.
    """

    initial_point_path = '/tmp/initial_point'
    model.saver.save(sess, initial_point_path, write_meta_graph=False)

    vars_to_monitor = []
    for v in model.var_list:
        if v.name.startswith(layer_name):
            print('added {} to monitor'.format(v.name))
            vars_to_monitor.append(v)

    default_graphdef = tf.get_default_graph().as_graph_def()
    layer_ops = [n.name for n in default_graphdef.node
                 if n.name.startswith(layer_name)]
    sub_graphdef = tf.graph_util.extract_sub_graph(default_graphdef, layer_ops)
    subgraph_ops = [n.name for n in sub_graphdef.node
                    if not n.name.startswith(layer_name)]

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_vars = [v for v in trainable_vars
                      if not v.name.split(':')[0] in subgraph_ops]

    gradients = []
    targets = []
    for idx in range(tasks.n):
        with tf.variable_scope(
                'optimizers/{}_optimizer'.format(tasks.names[idx])), \
                tf.control_dependencies(model.update_ops):
            optimizer = tf.train.MomentumOptimizer(
                    learning_rate=0.01, momentum=0.9, use_nesterov=True)
            loss = tasks.losses[idx].op
            grads_and_vars = optimizer.compute_gradients(
                    loss=loss, var_list=trainable_vars)
            gradients.append(grads_and_vars)
            targets.append(optimizer.apply_gradients(grads_and_vars))
    graph_utils.initialize_uninitialized_variables(sess)

    if steps_per_epoch is None:
        n_samples = np.array([f.n // f.batch_size for f in tasks.train_feeders])
    else:
        n_samples = np.array([steps_per_epoch for f in tasks.train_feeders])

    initial_vals = sess.run(vars_to_monitor)
    after_vals = [None] * tasks.n
    for task_id in range(tasks.n):
        model.saver.restore(sess, initial_point_path)
        batches = tqdm(range(n_samples[task_id]),
                       desc='task %s' % tasks.names[task_id])
        for batch in batches:
            x, y = next(tasks.train_feeders[task_id])
            ops_to_run = {'optimizer': targets[task_id]}
            feed_dict = {model.inputs[0]: x,
                         tasks.gt[task_id]: y,
                         K.learning_phase(): 1}
            sess.run(ops_to_run, feed_dict=feed_dict)
        after_vals[task_id] = sess.run(vars_to_monitor)

    print('\nDistances from initial state:')
    for task_id in range(tasks.n):
        print('||{} - init|| = '.format(tasks.names[task_id]), end=' ')
        for k in range(len(initial_vals)):
            after_vals[task_id][k] -= initial_vals[k]
        print([np.linalg.norm(after_vals[task_id][k])
               for k in range(len(initial_vals))])

    print('\nDistances between tasks:')
    for i in range(tasks.n):
        for j in range(i + 1, tasks.n):
            print('||{} - {}|| ='.format(
                tasks.names[i], tasks.names[j]), end=' ')
            print([np.linalg.norm(after_vals[j][k] - after_vals[i][k])
                   for k in range(len(initial_vals))])

    print('\nAngles between tasks:')
    for i in range(tasks.n):
        for j in range(i + 1, tasks.n):
            print('cos({}, {}) ='.format(
                tasks.names[i], tasks.names[j]), end=' ')
            print([np.divide((after_vals[j][k] * after_vals[i][k]).sum(),
                   (lg.norm(after_vals[j][k]) * lg.norm(after_vals[i][k])))
                   for k in range(len(initial_vals))])
