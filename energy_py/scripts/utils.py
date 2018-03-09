"""
A collection of helper functions.
"""
import csv
import logging
import pickle
import os

import tensorflow as tf


def ensure_dir(file_path):
    """
    Checks a directory exists.  If it doesn't - makes it.

    args
        file_path (str)
    """
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


def dump_pickle(obj, name):
    """
    Saves an object to a pickle file.

    args
        obj (object)
        name (str) path of the dumped file
    """
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    """
    Loads a an object from a pickle file.

    args
        name (str) path to file to be loaded

    returns
        obj (object)
    """
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)

    return obj


def make_logger(paths, name=None):
    """
    Sets up the energy_py logging stragety.  INFO to console, DEBUG to file.

    args
        paths (dict)
        name (str) optional logger name

    returns
        logger (object)
    """
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)

    fmt = '%(asctime)s [%(levelname)s]%(name)s: %(message)s'

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {'standard': {'format': fmt,
                                    'datefmt': '%Y-%m-%d %H:%M:%S'}},

        'handlers': {'console': {'level': 'INFO',
                                 'class': 'logging.StreamHandler',
                                 'formatter': 'standard'},

                     'debug_file': {'class': 'logging.FileHandler',
                                    'level': 'DEBUG',
                                    'filename': paths['debug_log'],
                                    'mode': 'w',
                                    'formatter': 'standard'},

                     'info_file': {'class': 'logging.FileHandler',
                                   'level': 'INFO',
                                   'filename': paths['info_log'],
                                   'mode': 'w',
                                   'formatter': 'standard'}},

        'loggers': {'': {'handlers': ['console', 'debug_file', 'info_file', ],
                         'level': 'DEBUG',
                         'propagate': True}}})

    return logger


def save_args(config, path):
    """
    Saves a config dictionary and optional argparse object to a text file.

    args
        config (dict)
        path (str) path for output text file
        argparse (object)

    returns
        writer (object) csv Writer object
    """
    with open(path, 'w') as outfile:
        writer = csv.writer(outfile)

        for k, v in config.items():
            print('{} : {}'.format(k, v))
            writer.writerow([k]+[v])

    return writer


class TensorboardHepler(object):
    """
    Holds a FileWriter and method for adding summaries

    args
        logdir (path)
    """
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)
        self.steps = 0

    def add_summaries(self, summaries):
        """
        Adds non-tensorflow data to tensorboard.

        args
            summaries (dict)
        """
        self.steps += 1
        for tag, var in summaries.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                         simple_value=var)])
            self.writer.add_summary(summary, self.steps)

        self.writer.flush()
