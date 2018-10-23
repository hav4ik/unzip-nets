import tensorflow as tf
import numpy as np
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
                   task_names,
                   losses,
                   true_tensors,
                   train_feeders,
                   val_feeders,
                   out_dir):

    train_samples = np.array([f.n // f.batch_size for f in train_feeders])
    val_samples = np.array([f.n // f.batch_size for f in val_feeders])

    initial_point_path = '/tmp/initial_point'
    model.saver.save(sess, initial_point_path, write_meta_graph=False)

    vars_to_monitor = []
    subgraph_sets = []
    for v in model.var_list:
        if v.name.startswith(layer_name):
            print('added {} to monitor'.format(v.name))
            vars_to_monitor.append(v)

    default_graphdef = tf.get_default_graph().as_graph_def()
    layer_ops = [n.name for n in default_graphdef.node if n.name.startswith(layer_name)]
    sub_graphdef = tf.graph_util.extract_sub_graph(default_graphdef, layer_ops)
    subgraph_ops = [n.name for n in sub_graphdef.node if not n.name.startswith(layer_name)]

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_vars = [v for v in trainable_vars if not v.name.split(':')[0] in subgraph_ops]

    gradients = []
    targets = []
    for idx in range(len(model.outputs)):
        with tf.variable_scope(
                'optimizers/{}_optimizer'.format(task_names[idx])), \
                tf.control_dependencies(model.update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True)
            loss = losses[idx].op
            grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=trainable_vars)
            gradients.append(grads_and_vars)
            targets.append(optimizer.apply_gradients(grads_and_vars))
    graph_utils.initialize_uninitialized_variables(sess)

    n_samples = np.array([f.n // f.batch_size for f in train_feeders])
    initial_vals = sess.run(vars_to_monitor)
    after_vals = [None] * len(model.outputs)
    for task_id in range(len(model.outputs)):
        model.saver.restore(sess, initial_point_path)
        batches = tqdm(range(n_samples[task_id]), desc='task %s' % task_names[task_id])
        for batch in batches:
            x, y = next(train_feeders[task_id])
            ops_to_run = {'optimizer': targets[task_id]}
            feed_dict = {model.inputs[0]: x,
                         true_tensors[task_id]: y,
                         K.learning_phase(): 1}
            sess.run(ops_to_run, feed_dict=feed_dict)
        after_vals[task_id] = sess.run(vars_to_monitor)

    for task_id in range(len(model.outputs)):
        print('||{} - init|| = '.format(task_names[task_id]), end=' ')
        print([np.linalg.norm(after_vals[task_id][k] - initial_vals[k]) for k in range(len(initial_vals))])
    for i in range(len(model.outputs)):
        for j in range(i + 1, len(model.outputs)):
            print('||{} - {}|| ='.format(task_names[i], task_names[j]), end=' ')
            print([np.linalg.norm(after_vals[j][k] - after_vals[i][k]) for k in range(len(initial_vals))])


def grad_stats(sess,
               model,
               var_names,
               task_names,
               losses,
               true_tensors,
               train_feeders,
               val_feeders,
               out_dir):

    train_samples = np.array([f.n // f.batch_size for f in train_feeders])
    val_samples = np.array([f.n // f.batch_size for f in val_feeders])

    gradients = []
    targets = []
    for idx in range(len(model.outputs)):
        with tf.variable_scope(
                'optimizers/{}_optimizer'.format(task_names[idx])), \
                tf.control_dependencies(model.update_ops):
            optimizer = tf.train.GradientDescentOptimizer(0.001)
            loss = losses[idx].op
            grads_and_vars = optimizer.compute_gradients(loss=loss)
            gradients.append(grads_and_vars)
            targets.append(optimizer.apply_gradients(grads_and_vars))
    graph_utils.initialize_uninitialized_variables(sess)

    var_names = var_names.split(',')
    var_grads = []
    var_saves = []
    for g, v in gradients[0]:
        if g is None:
            continue
        if not v.name in var_names:
            continue

    train_pb = tqdm(range(train_samples[0]))

    for i in train_pb:
        x, y = next(train_feeders[0])
        ops_to_run = {'var_grads': var_grads}
        feed_dict = {model.inputs[0]: x,
                     true_tensors[0]: y,
                     K.learning_phase(): 0}
        out = sess.run(ops_to_run, feed_dict=feed_dict)





