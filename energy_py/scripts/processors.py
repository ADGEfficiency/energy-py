"""
Objects to preprocess numpy arrays

Module contains
    Normalizer
    Standardizer
"""

import numpy as np


#  use epsilon to catch div/0 errors
epsilon = 1e-5


class ProcessSigmoid():

    def __init__(self):
        pass

    def transform(self, x):
        return 1 / (1 + np.exp(-x))


class ProcessTanh():
    def __init__(self):
        pass

    def transform(self, x):
        return np.tanh(x)


class Normalizer(object):
    """
    Normalization to range [0, 1]

    normalized = (value - minimum) / (maximum - minimum)

    args
        array (np.array)
    """

    def __init__(self, use_history=False):
        self.shape = None
        self.mins = None
        self.maxs = None
        self.use_history = use_history

    def transform(self, batch):
        if self.use_history:
            return self.transform_hist(batch)

        else:
            mins = np.min(batch)
            maxs = np.max(batch)
            return (batch - mins) / (maxs - mins + epsilon)

    def transform_hist(self, batch):
        """
        Normalizes an array.

        args
            batch (np.array)

        returns
            transformed (np.array)
        """
        assert batch.ndim == 2

        #  catching the unitialized processor
        if self.mins is None:
            self.shape = batch.shape[1:]
            self.mins = batch.min(axis=0).reshape(1, *self.shape)
            self.maxs = batch.max(axis=0).reshape(1, *self.shape)

        hist = np.concatenate([self.mins,
                               self.maxs,
                               batch]).reshape(-1, *self.shape)

        self.mins = hist.min(axis=0).reshape(1, *self.shape)
        self.maxs = hist.max(axis=0).reshape(1, *self.shape)

        return (batch - self.mins) / (self.maxs - self.mins + epsilon)


class Standardizer(object):
    """
    Processor object for performing standardization
    Standardization = scaling for zero mean, unit variance

    Algorithm from post by Dinesh 2011 on Stack Exchange:
    https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values

    Statistics are calculated online, without keeping entire history (ie each batch)

    Idea is to keep three running counters
        sum(x)
        sum(x^2)
        N (count)

    We can then calculate historical statistics by:
        mean = sum(x) / N
        variance = 1/N * [sum(x^2) - sum(x)^2 / N]
        standard deviation = sqrt(variance)

    """
    def __init__(self):

        #  initial stats
        self.shape = None
        self.count = 0
        self.sum = None
        self.sum_sq = None
        self.means = None
        self.stds = None

    def transform(self, batch):
        assert batch.ndim == 2
        if self.shape is None:
            self.shape = batch.shape[1:]
            self.sum = np.zeros(shape=(1, *self.shape))
            self.sum_sq = np.zeros(shape=(1, *self.shape))

        #  update our three counters
        self.sum = np.sum(np.concatenate([self.sum, batch]),
                          axis=0).reshape(1, *self.shape)
        self.sum_sq = np.sum(np.concatenate([self.sum_sq, batch**2]),
                             axis=0).reshape(1, *self.shape)
        self.count += batch.shape[0]

        #  calculate the mean, variance and standard deivation
        self.means = self.sum / self.count
        var = (self.sum_sq - self.sum**2 / self.count) / self.count
        self.stds = np.sqrt(var)

        #  perform the de-meaning & scaling by standard deivation
        return (batch - self.means) / (self.stds + epsilon)
