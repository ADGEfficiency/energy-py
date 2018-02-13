"""
A collection of helper functions.
"""
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


def make_logger(log_path, name=None):

    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)

    logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,

            'formatters': {'standard': {'format': '%(asctime)s [%(levelname)s]%(name)s: %(message)s'}},

            'handlers': {'console': {'level': 'INFO',
                                     'class': 'logging.StreamHandler',
                                     'formatter': 'standard'},

                         'file': {'class': 'logging.FileHandler',
                                  'level': 'DEBUG',
                                  'filename': log_path,
                                  'mode': 'w',
                                  'formatter': 'standard'}, },

            'loggers': {'': {'handlers': ['console', 'file', ],
                             'level': 'DEBUG',
                             'propagate': True}}})

    return logger


class TensorboardHepler(object):

    def __init__(self, logdir):

        self.writer = tf.summary.FileWriter(logdir)
        self.steps = 0

    def add_summaries(self, summaries):
        self.steps += 1
        for tag, var in summaries.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                         simple_value=var)])
            self.writer.add_summary(summary, self.steps)

        self.writer.flush()
