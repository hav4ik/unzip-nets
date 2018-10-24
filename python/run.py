from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import config_parser
from utils.config_parser import import_feeders_from_cfg
from utils.config_parser import import_losses_from_cfg
from utils.config_parser import import_metrics_from_cfg
from utils.config_parser import import_optimizers_from_cfg

from unzipping.train import train_alternating
from unzipping.stats import integral_stats


def _prepare_environment():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    return sess


def _prepare_feeder_placeholders(outputs, tasks_names):
    with tf.variable_scope('feeders'):
        ground_truths = []
        for idx in range(len(outputs)):
            y = tf.placeholder(
                    shape=outputs[idx].get_shape(), dtype=tf.float32,
                    name='{}_feeder'.format(tasks_names[idx]))
            ground_truths.append(y)
    return ground_truths


def run_app(app_name,
            experiment_config,
            batch_size,
            weights,
            out_dir,
            app_args):
    """Run an applications from our unzipping toolbox
    """
    sess = _prepare_environment()

    experiment_name, cfg = config_parser.read_config(experiment_config)

    task_names = config_parser.get_tasks_info(cfg)
    model = config_parser.import_model_from_cfg(cfg, sess)
    if weights is not None:
        model.saver.restore(sess, weights)

    ground_truths = _prepare_feeder_placeholders(model.outputs, task_names)
    train_feeders, val_feeders = import_feeders_from_cfg(cfg, batch_size)
    optimizer_defs = import_optimizers_from_cfg(cfg)

    losses = import_losses_from_cfg(cfg, model, ground_truths, task_names)
    metrics = import_metrics_from_cfg(cfg, model, ground_truths, task_names)

    if app_name == 'train':
        train_alternating(
                experiment_name, task_names, sess, model, losses, metrics,
                optimizer_defs, ground_truths, train_feeders, val_feeders,
                out_dir, args.epochs)

    elif app_name == 'ipaths':
        integral_stats(
                sess, model, args.var_name, task_names, losses, ground_truths,
                train_feeders, val_feeders, out_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('app', type=str, help='Unzipping tool to run')
    parser.add_argument(
            'config', type=str, help='A JSON file or a string in JSON format')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-o', '--out_dir', type=str, default='out/')
    parser.add_argument('-w', '--weights', type=str)

    train_args = parser.add_argument_group('train')
    train_args.add_argument('-n', '--epochs', type=int, default=10)

    gradstat_args = parser.add_argument_group('grad stat')
    gradstat_args.add_argument('-v', '--var_name', type=str)

    args = parser.parse_args()
    run_app(args.app,
            args.config,
            args.batch_size,
            args.weights,
            args.out_dir,
            args)
