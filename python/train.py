import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import config_parser
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


def _prepare_dirs(cfg, out_dir):
    """Prepares the output directory structure; the directory is
       uniquely numbered.
    """
    tensorboard_dir = os.path.join(
            os.path.expanduser(out_dir), 'tensorboard', cfg['experiment_name'])
    checkpoints_dir = os.path.join(
            os.path.expanduser(out_dir), 'checkpoints', cfg['experiment_name'])

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


def _make_summaries(outputs, metrics_avg, loss_avg, tasks_names):
    train_epoch_summary_list = []
    for idx in range(len(outputs)):
        for m_idx in range(len(metrics_avg[idx])):
            train_epoch_summary_list.append(tf.summary.scalar(
                'train/{}/metrics_{}'.format(tasks_names[idx], m_idx),
                metrics_avg[idx][m_idx]))
        train_epoch_summary_list.append(tf.summary.scalar(
            'train/{}/loss'.format(tasks_names[idx]), loss_avg[idx]))
    train_epoch_summary = tf.summary.merge(train_epoch_summary_list)

    val_epoch_summary_list = []
    for idx in range(len(outputs)):
        for m_idx in range(len(metrics_avg[idx])):
            val_epoch_summary_list.append(tf.summary.scalar(
                'val/{}/metrics_{}'.format(tasks_names[idx], m_idx),
                metrics_avg[idx][m_idx]))
        val_epoch_summary_list.append(tf.summary.scalar(
            'val/{}/loss'.format(tasks_names[idx]), loss_avg[idx]))
    val_epoch_summary = tf.summary.merge(val_epoch_summary_list)

    return train_epoch_summary, val_epoch_summary


def _prepare_feeder_placeholders(outputs, tasks_names):
    with tf.variable_scope('feeders'):
        ground_truths = []
        for idx in range(len(outputs)):
            y = tf.placeholder(
                    shape=outputs[idx].get_shape(), dtype=tf.float32,
                    name='{}_feeder'.format(tasks_names[idx]))
            ground_truths.append(y)
    return ground_truths


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
    tensorboard_dir, checkpoints_dir = _prepare_dirs(cfg, out_dir)

    tasks_names = config_parser.get_tasks_info(cfg)
    inputs, outputs, update_ops, regularizer, model_saver = \
        config_parser.import_model_from_cfg(sess, cfg)

    ground_truths = _prepare_feeder_placeholders(outputs, tasks_names)
    training_feeders, validating_feeders = \
        config_parser.import_feeders_from_cfg(cfg, batch_size)
    optimizer_op, optimizer_params, lr_p, lr_s = \
        config_parser.import_optimizers_from_cfg(cfg)

    losses, loss_accum, loss_reset, loss_avg = \
        config_parser.import_losses_from_cfg(
                cfg, outputs, ground_truths, regularizer, tasks_names)
    metrics, metrics_accum, metrics_reset, metrics_avg = \
        config_parser.import_metrics_from_cfg(
                cfg, outputs, ground_truths, tasks_names)
    loss_and_metric_reset_ops = (loss_reset, metrics_reset)
    loss_and_metric_avg_ops = (loss_avg, metrics_avg)

    train_feeder_idx, train_samples = _concatenate_feeders(training_feeders)
    val_feeder_idx, val_samples = _concatenate_feeders(validating_feeders)

    if mode == 'alternating':
        print('\nStarted training in mode "Alternating"')
        print('checkpoints: {}'.format(checkpoints_dir))
        print('logs: {}'.format(tensorboard_dir))

        gradients = []
        targets = []
        for idx in range(len(outputs)):
            with tf.variable_scope(
                    'optimizers/{}_optimizer'.format(tasks_names[idx])), \
                    tf.control_dependencies(update_ops):
                optimizer = optimizer_op(**optimizer_params)
                loss = losses[idx]
                grads_and_vars = optimizer.compute_gradients(loss=loss)
                gradients.append(grads_and_vars)
                targets.append(optimizer.apply_gradients(grads_and_vars))
        graph_utils.initialize_uninitialized_variables(sess)

        with tf.variable_scope('summaries'):
            train_epoch_summary, val_epoch_summary = _make_summaries(
                    outputs, metrics_avg, loss_avg, tasks_names)
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
            sess.run(loss_and_metric_reset_ops)
            for idx in train_idxs:
                x, y = next(training_feeders[idx])
                ops_to_run = {'loss_accumulate': loss_accum[idx],
                              'metrics_accumulate': metrics_accum[idx],
                              'optimizer': targets[idx]}
                feed_dict = {inputs[0]: x,
                             ground_truths[idx]: y,
                             K.learning_phase(): 1}
                if lr_p is not None and lr_s is not None:
                    feed_dict.update({lr_p: current_lr})
                sess.run(ops_to_run, feed_dict=feed_dict)
            out = sess.run({
                'loss_n_metrics': loss_and_metric_avg_ops,
                'train_summary': train_epoch_summary})
            trn_losses, trn_metrics = out['loss_n_metrics']
            summary_writer.add_summary(out['train_summary'], epoch)

            val_idxs = tqdm(val_feeder_idx[:], desc='val')
            sess.run(loss_and_metric_reset_ops)
            for idx in val_idxs:
                x, y = next(validating_feeders[idx])
                ops_to_run = {'loss_accumulate': loss_accum[idx],
                              'metrics_accumulate': metrics_accum[idx]}
                feed_dict = {inputs[0]: x,
                             ground_truths[idx]: y,
                             K.learning_phase(): 0}
                sess.run(ops_to_run, feed_dict=feed_dict)
            out = sess.run({
                'hist_summary': histogram_summary,
                'loss_n_metrics': loss_and_metric_avg_ops,
                'val_summary': val_epoch_summary})
            val_losses, val_metrics = out['loss_n_metrics']
            summary_writer.add_summary(out['val_summary'], epoch)
            summary_writer.add_summary(out['hist_summary'], epoch)

            for i in range(len(outputs)):
                print('task #{}:'.format(i))
                print('  -(train): loss={:3.6f}, metrics=[{}]'.format(
                    trn_losses[i],
                    ', '.join(['{:.6f}'.format(x) for x in trn_metrics[i]])))
                print('  -(val)  : loss={:3.6f}, metrics=[{}]'.format(
                    val_losses[i],
                    ', '.join(['{:.6f}'.format(x) for x in val_metrics[i]])))

            model_saver.save(
                    sess, os.path.join(checkpoints_dir, 'epoch'),
                    write_meta_graph=False, global_step=epoch)

        model_saver.save(sess, os.path.join(checkpoints_dir, 'final_model'))

    else:
        raise NameError('Training mode "{}" not supported yet.'.format(mode))


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
