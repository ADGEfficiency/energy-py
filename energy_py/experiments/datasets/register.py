"""
A registry for datasets

All this registry holds is paths
"""

import logging
import os

logger = logging.getLogger(__name__)


def make_registry():
    """
    Creates a registry of datasets

    returns
        registry (dict) {dataset_name: path/to/dataset}

    Bug where __pychache__ will be added to the register
    Could be fixed by checking for the existence of state.csv
    """
    #  get the current directory of the datasets folder
    dirname = os.path.dirname(os.path.abspath(__file__))

    #  get a list of directories and files in the datasets folder
    dirs = os.listdir(dirname)

    registry = {}

    for dataset_name in dirs:
        directory = os.path.join(dirname, dataset_name)

        if os.path.isdir(directory):
            registry[dataset_name] = directory

    return registry


registry = make_registry()


def get_dataset_path(dataset_name):

    logger.info('Getting dataset {} path'.format(dataset_name))

    return registry[dataset_name]
