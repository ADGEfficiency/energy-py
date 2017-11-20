"""
Purpose of this script is to hold Processor objects - a class of objects
used for functionality such as standardization or normalization.

The idea behind using a class is that we can allow the class to collect
statistics on the data it has seen so far - for example to estimate the mean
for standardization from all data the Processor has processed so far.
"""

import numpy as np

class SimpleStandardizer(object):
    def __init__(self):
        pass

    def transform(self, data):
        """
        The SimpleStandardizer transforms the data using the mean and standard
        deviation across the batch it is processing
        """
        return (data - data.mean()) / data.std()

class Standardizer(object):
    """
    Holds all historical data
    """
    def __init__(self):
        self.data = np.array([])

    def transform(self, data):
        """
        Transform the batch using the avergage mean & std we have seen over
        the lifetime of the Standardizer
        """
        self.data = np.append(self.data, data)
        means = np.mean(self.data, axis=1)
        std = np.std(self.data, axis=1)
        return (data - mean) / std

class SimpleNormalizer(object):
    def __init__(self):
        pass

    def transform(self, data):
        """
        Transforms the batch using the largest & smallest
        """
        maximum = np.maximum(data)
        minimum = np.minimum(data)
        return (data  - minimum) / (maximum - minimum)

class Normalizer(object):
    def __init__(self):
        self.data = np.array([])

    def transform(self, data):
        """
        Transforms the batch using the largest & smallest
        """
        self.data = np.append(self.data, data)
        maximum = np.maximum(self.data, axis=1)
        minimum = np.minimum(self.data, axis=1)
        return (data  - minimum) / (maximum - minimum)
