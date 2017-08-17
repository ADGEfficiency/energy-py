"""
A collection of helper functions
"""

import pickle
import os
import time

def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return None


def get_upper_path(string):
    owd = os.getcwd()  #  save original working directory
    os.chdir(string)  #  move back two directories
    base = os.getcwd()  #  get new wd
    os.chdir(owd)  #  reset wd
    return base

