import numpy as np


class DiscreteSpace(object):
    """
    A single dimension discrete space

    args
        num (int) the number of options across the discrete space
    """

    def __init__(self, num):
        self.low = 0
        self.high = num

    def sample(self):
        return np.random.randint(self.high)

    def contains(self, x):
        return np.in1d(x, np.arange(0, self.high))

    def discretize(self, n_discr=None):
        return np.arange(0, self.high + 1)
