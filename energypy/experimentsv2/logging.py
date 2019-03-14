import logging.config
from os.path import join
import sys


# def make_new_logger(log_dir, name):
#     logger = logging.getLogger(name)

#     fmt = '%(asctime)s [%(levelname)s]%(name)s: %(message)s'

#     logging.config.dictConfig(
#         {
#             'version': 1,
#             'disable_existing_loggers': False,

#             'formatters': {
#                 'standard': {'format': fmt, 'datefmt': '%Y-%m-%dT%H:M:S'},
#                 'file': {'format': '%(message)s'}
#             },

#             'handlers': {
#                 'console': {
#                     'level': 'INFO',
#                     'class': 'logging.StreamHandler',
#                     'formatter': 'standard'},

#                 'file': {
#                      'class': 'logging.FileHandler',
#                      'level': 'DEBUG',
#                      'filename': join(log_dir, '{}.log'.format(name)),
#                      'mode': 'w',
#                      'formatter': 'file'}
#             },

#             'loggers': {
#                 '': {
#                   'handlers': ['console', 'file'],
#                   'level': 'DEBUG',
#                   'propagate': True}
#             }
#         }
#     )

    # return logger
import logging
formatter = logging.Formatter('%(message)s')


def make_new_logger(name, log_dir):
    """Function setup as many loggers as you want"""
    log_file = join(log_dir, name+'.log')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(stream)

    return logger
