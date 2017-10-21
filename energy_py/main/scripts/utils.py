import itertools
import pickle
import os
import time

import numpy as np


class Utils(object):
    """
    A base class that holds generic functions
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    """
    Useful Python functions:
    """

    def verbose_print(self, *args):
        """
        Helper function to print info.
        """
        if self.verbose:
            [print(a) for a in args]
        return None

    def dump_pickle(self, obj, name):
        with open(name, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, name):
        with open(name, 'rb') as handle:
            obj = pickle.load(handle)
        return obj

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return None

    def get_upper_path(self, string):
        owd = os.getcwd()  #  save original working directory
        os.chdir(string)  #  move back two directories
        base = os.getcwd()  #  get new wd
        os.chdir(owd)  #  reset wd
        return base

    """
    energy_py specific generic functions:
    """

    def normalize(self, value, low, high):
        """
        Generic helper function
        Normalizes a value using a given lower & upper bound
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
        Default scaler is to normalize

        Used to scale the observation and action

        args
            array : np array (1, space_length)
            space : list len(action or observation space)

        returns
            scaled_array : np array (1, space_length)
        """

        #  empty numpy array
        scaled_array = np.array([])
        array = array.reshape(-1)
        assert array.shape[0] == len(space)

        #  iterate across the array values & corresponding space object
        for value, spc in itertools.zip_longest(array, space):
            if spc.type == 'continuous':
                # normalize continuous variables
                scaled = self.normalize(value, spc.low, spc.high)
            elif spc.type == 'discrete':
                #  shouldn't need to do anything
                #  check value is already dummy
                assert (value == 0) or (value == 1)
            else:
                assert 1 == 0

            #  appending the scaled value onto the scaled array
            scaled_array = np.append(scaled_array, scaled)

        return scaled_array.reshape(1, len(space))
