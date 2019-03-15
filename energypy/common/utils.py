import json
import os
import pickle




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
        return pickle.load(handle)


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def dump_config(cfg, logger):
    for k, v in cfg.items():
        logger.info(json.dumps({k: v}))

def read_iterable_from_config(argument):
    if isinstance(argument, str):
        argument = argument.split(',')
        argument = [int(argument) for argument in argument]
    else:
        argument = tuple(argument)

    return argument
