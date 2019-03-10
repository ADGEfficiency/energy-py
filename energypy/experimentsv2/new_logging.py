import json
import logging
import logging.config


def read_logs(log_file_path):
    with open(log_file_path) as f:
        logs = f.read().splitlines()

    return [json.loads(log) for log in logs]


def make_logger():
    logger = logging.getLogger(__name__)

    fmt = "%(asctime)-15s %(levelname)-8s %(message)s"

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {'standard': {'format': fmt, 'datefmt': '%Y-%m-%d'},
                       'file': {'format': '%(message)s'}},

        'handlers': {'console': {'level': 'INFO',
                                 'class': 'logging.StreamHandler',
                                 'formatter': 'standard'},

                     'file': {'class': 'logging.FileHandler',
                              'level': 'INFO',
                              'filename': './log.log',
                              'mode': 'w',
                              'formatter': 'standard'}},

        'loggers': {'': {'handlers': ['console', 'file'],
                         'level': 'DEBUG',
                         'propagate': True}}})

    return logger

logger.info(json.dumps(dict))
