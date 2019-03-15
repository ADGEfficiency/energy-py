import logging
import logging.config
from os.path import join
import sys


formatter = logging.Formatter('%(message)s')


def make_new_logger(name, log_dir=None):
    """Function setup as many loggers as you want"""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if log_dir:
        log_file = join(log_dir, name+'.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    logger.addHandler(stream)

    return logger
