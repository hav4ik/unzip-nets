import os
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
from tqdm import tqdm

from utils import graph_utils


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


def _l2_filter_norm(t):
    return tf.sqrt(tf.reduce_sum(
        tf.pow(t, 2),
        np.arange(len(t.get_shape()) - 1, dtype=np.int32)))


def _make_summaries(metrics, losses, tasks_names):
    assert len(metrics) == len(losses)
    train_epoch_summary_list = []
    for idx in range(len(metrics)):
        for m_idx in range(len(metrics[idx])):
            train_epoch_summary_list.append(tf.summary.scalar(
                'train/{}/metrics_{}'.format(tasks_names[idx], m_idx),
                metrics[idx][m_idx].average))
        train_epoch_summary_list.append(tf.summary.scalar(
            'train/{}/loss'.format(tasks_names[idx]), losses[idx].average))
    train_epoch_summary = tf.summary.merge(train_epoch_summary_list)

    val_epoch_summary_list = []
    for idx in range(len(metrics)):
        for m_idx in range(len(metrics[idx])):
            val_epoch_summary_list.append(tf.summary.scalar(
                'val/{}/metrics_{}'.format(tasks_names[idx], m_idx),
                metrics[idx][m_idx].average))
        val_epoch_summary_list.append(tf.summary.scalar(
            'val/{}/loss'.format(tasks_names[idx]), losses[idx].average))
    val_epoch_summary = tf.summary.merge(val_epoch_summary_list)

    return train_epoch_summary, val_epoch_summary


def _prepare_dirs(experiment_name, out_dir):
    """Prepares the output directory structure; the directory is
       uniquely numbered.
    """
    tensorboard_dir = os.path.join(
            os.path.expanduser(out_dir), 'tensorboard', experiment_name)
    checkpoints_dir = os.path.join(
            os.path.expanduser(out_dir), 'checkpoints', experiment_name)

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


def train_alternating(sess,
                      exp_name,
                      model,
                      tasks,
                      optimizer_defs,
                      out_dir,
                      n_epochs):

    train_feeder_idx, train_samples = _concatenate_feeders(tasks.train_feeders)
    val_feeder_idx, val_samples = _concatenate_feeders(tasks.val_feeders)

    tensorboard_dir, checkpoints_dir = _prepare_dirs(exp_name, out_dir)
    optimizer_op, optimizer_params, lr_p, lr_s = optimizer_defs

    loss_reset_ops = [l.resetter for l in tasks.losses]
    loss_avg_ops = [l.average for l in tasks.losses]
    metric_reset_ops = [[m.resetter for m in t] for t in tasks.metrics]
    metric_avg_ops = [[m.average for m in t] for t in tasks.metrics]
    metric_accumulators = [[m.accumulator for m in t] for t in tasks.metrics]

    print('\nStarted training')
    print('checkpoints: {}'.format(checkpoints_dir))
    print('logs: {}'.format(tensorboard_dir))

    gradients = []
    targets = []
    for idx in range(tasks.n):
        with tf.variable_scope(
                'optimizers/{}_optimizer'.format(tasks.names[idx])), \
                tf.control_dependencies(model.update_ops):
            optimizer = optimizer_op(**optimizer_params)
            loss = tasks.losses[idx].op
            grads_and_vars = optimizer.compute_gradients(loss=loss)
            gradients.append(grads_and_vars)
            targets.append(optimizer.apply_gradients(grads_and_vars))
    graph_utils.initialize_uninitialized_variables(sess)

    with tf.variable_scope('summaries'):
        train_epoch_summary, val_epoch_summary = \
                _make_summaries(tasks.metrics, tasks.losses, tasks.names)
        histogram_summary_list = []
        for g, v in gradients[0]:
            if g is None:
                continue
            histogram_summary_list.append(tf.summary.histogram(
                "variables/" + v.name, _l2_filter_norm(v)))
        histogram_summary = tf.summary.merge(histogram_summary_list)
    summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

    for epoch in range(n_epochs):
        if lr_s is not None:
            current_lr = lr_s(epoch)
        else:
            current_lr = 'Auto'
        print('\nEpoch #{}: (lr={})'.format(epoch, current_lr))

        np.random.shuffle(train_feeder_idx)
        train_idxs = tqdm(train_feeder_idx[:], desc='trn')
        sess.run([loss_reset_ops, metric_reset_ops])
        for idx in train_idxs:
            x, y = next(tasks.train_feeders[idx])
            ops_to_run = {'loss_accumulate': tasks.losses[idx].accumulator,
                          'metrics_accumulate': metric_accumulators[idx],
                          'optimizer': targets[idx]}
            feed_dict = {model.inputs[0]: x,
                         tasks.gt[idx]: y,
                         K.learning_phase(): 1}
            if lr_p is not None and lr_s is not None:
                feed_dict.update({lr_p: current_lr})
            sess.run(ops_to_run, feed_dict=feed_dict)
        out = sess.run({
            'loss_n_metrics': (loss_avg_ops, metric_avg_ops),
            'train_summary': train_epoch_summary})
        trn_losses, trn_metrics = out['loss_n_metrics']
        summary_writer.add_summary(out['train_summary'], epoch)

        val_idxs = tqdm(val_feeder_idx[:], desc='val')
        sess.run(metric_reset_ops)
        for idx in val_idxs:
            x, y = next(tasks.val_feeders[idx])
            ops_to_run = {'loss_accumulate': tasks.losses[idx].accumulator,
                          'metrics_accumulate': metric_accumulators[idx]}
            feed_dict = {model.inputs[0]: x,
                         tasks.gt[idx]: y,
                         K.learning_phase(): 0}
            sess.run(ops_to_run, feed_dict=feed_dict)
        out = sess.run({
            'hist_summary': histogram_summary,
            'loss_n_metrics': (loss_avg_ops, metric_avg_ops),
            'val_summary': val_epoch_summary})
        val_losses, val_metrics = out['loss_n_metrics']
        summary_writer.add_summary(out['val_summary'], epoch)
        summary_writer.add_summary(out['hist_summary'], epoch)

        for i in range(tasks.n):
            print('task "{}":'.format(tasks.names[i]))
            print('  -(train): loss={:3.6f}, metrics=[{}]'.format(
                trn_losses[i],
                ', '.join(['{:.6f}'.format(x) for x in trn_metrics[i]])))
            print('  -(val)  : loss={:3.6f}, metrics=[{}]'.format(
                val_losses[i],
                ', '.join(['{:.6f}'.format(x) for x in val_metrics[i]])))

        model.saver.save(
                sess, os.path.join(checkpoints_dir, 'epoch'),
                write_meta_graph=False, global_step=epoch)

    model.saver.save(sess, os.path.join(checkpoints_dir, 'final_model'))
