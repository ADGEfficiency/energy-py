import csv
from itertools import zip_longest
import pickle
import os
import time

import numpy as np


class Utils(object):
    """
    A base class that holds generic functions
    """
    def __init__(self):
        pass

    """
    Useful Python functions:
    """

    @staticmethod
    def dump_pickle(obj, name):
        with open(name, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(name):
        with open(name, 'rb') as handle:
            obj = pickle.load(handle)
        return obj

    @staticmethod
    def ensure_dir(file_path):
        """
        Check that a directory exists
        If it doesn't - make it
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def get_upper_path(string):
        owd = os.getcwd()  #  save original working directory
        os.chdir(string)  #  move based on the input string
        base = os.getcwd()  #  get new wd
        os.chdir(owd)  #  reset wd
        return base

    @staticmethod
    def save_args(argparse, path, optional={}):
        """
        Saves args from an argparse object and from an optional
        dictionary

        args
            argparse (object)
            path (str)        : path to save too
            optional (dict)   : optional dictionary of additional arguments

        returns
            writer (object) : csv Writer object
        """
        with open(path, 'w') as outfile:
            writer = csv.writer(outfile)
            for k, v in vars(argparse).items():
                print('{} : {}'.format(k, v))
                writer.writerow([k]+[v])

            if optional:
                for k, v in optional.items():
                    print('{} : {}'.format(k, v))
                    writer.writerow([k]+[v])
        return writer

    """
    energy_py specific functions
    """

    @staticmethod
    def normalize(value, low, high):
        """
        Generic helper function
        Normalizes a value using a given lower & upper bound

        args
            value (float)
            low   (float) : upper bound
            high  (float) : lower_bound

        returns
            normalized (np array)
        """
        #  if statement to catch the constant value case
        if low == high:
            normalized = 0
        else:
            max_range = high - low
            normalized = (value - low) / max_range
        return np.array(normalized)

    def scale_array(self, array, space):
        """
        Helper function for make_machine_experience()
        Uses the space & a given function to scale an array
        Scaling is done by normalization

        Used to scale the observations and actions

        Only really setup for continuous spaces TODO

        args
            array (np array) : array to be scaled
                               shape=(1, space_length)

            space (list) : a list of energy_py Space objects
                           shape=len(action or observation space)

        returns
            scaled_array (np array)  : the scaled array
                                       shape=(1, space_length)
        """
        array = array.reshape(-1)
        assert array.shape[0] == space.shape[0]

        scaled_array = np.array([])
        for low, high, val in zip_longest(space.low, space.high, array):
           scaled = self.normalize(low, high, val) 
           scaled_array = np.append(scaled_array, scaled)

        return scaled_array.reshape(1, space.shape[0])
