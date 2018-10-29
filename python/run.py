from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import config_parser
from utils.config_parser import import_optimizers_from_cfg

from unzipping.train import train_alternating
from unzipping.stats import integral_stats


def _prepare_environment():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    return sess


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
    tasks = config_parser.Tasks(cfg)

    model = config_parser.import_model_from_cfg(sess, tasks.names, cfg)
    if weights is not None:
        model.saver.restore(sess, weights)

    tasks.wrap_on_model(model)
    tasks.load_feeders(batch_size)
    optimizer_defs = import_optimizers_from_cfg(cfg)

    if app_name == 'train':
        train_alternating(sess,
                          experiment_name,
                          model,
                          tasks,
                          optimizer_defs,
                          out_dir,
                          args.epochs)

    elif app_name == 'ipaths':
        integral_stats(sess,
                       model,
                       args.var_name,
                       tasks,
                       out_dir)


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
