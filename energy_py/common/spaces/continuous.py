import numpy as np


class ContinuousSpace(object):
    """
    A single dimension continuous space

    args
        low  (float) minimum bound for a single dimension
        high (float) maximum bound for a single dimension
    """

    def __init__(self, low, high):
        self.low  = float(low)
        self.high = float(high)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        return (x >= self.low) and (x <= self.high)

    def discretize(self, n_discr):
        return np.linspace(self.low, self.high, n_discr).tolist()


class ImageSpace(object):
    """
    2D image
    """

    def __init__(self, low, high, shape):
        self.low  = float(low)
        self.high = float(high)
        self.shape = shape

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()


