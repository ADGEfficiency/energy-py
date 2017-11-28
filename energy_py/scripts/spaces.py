"""
This module holds energy_py Space objects.

Inspired by the OpenAI gym spaces.

Compatability with gym spaces is ideal as it allows enegy_py agents
to be used with gym environments.

The energy_py GlobalSpace is the equivilant of the gym TupleSpace.
"""

import numpy as np


class DiscreteSpace(object):
    """
    A single dimension discrete space.
    - an on/off switch
    - a single button on a keyboard

    Args:
        low  (float) : an array with the minimum bound for each
        high (float) : an array with the maximum bound for each
        step (float) : an array with step size
    """

    def __init__(self, low, high):
        self.low  = int(low)
        self.high = int(high)
        self.type = 'discrete'
        self.discrete_space = np.arange(self.low, self.high + 1) 

    def sample(self):
        return np.random.choice(self.discrete_space)

    def contains(self, x):
        return np.in1d(x, self.discrete_space)


class ContinuousSpace(object):
    """
    A single dimension continuous space.
    - a car accelerator
    - load on a gas turbine
    - speed of a variable speed drive

    Args:
        low  (float) : an array with the minimum bound for each
        high (float) : an array with the maximum bound for each
    """

    def __init__(self, low, high):
        self.low  = float(low)
        self.high = float(high)
        self.type = 'continuous'

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        return (x >= self.low) and (x <= self.high)

    def discretize(self, num_discrete):
        return np.linspace(self.low, self.high, num_discrete)


class GlobalSpace(object):
    """
    A combination of multiple spaces
    All energy_py environments use this as the observation.space
    or action.space object

    Similar to the OpenAI gym Tuple space object

    """
    def __init__(self, spaces):
        #  our space is a tuple of the simpler spaces
        self.spaces = [spc for spc in spaces]
        self.length = len(self.spaces)
        self.type = 'global'

        self.shape = self._get_shape()
        self.low = self._get_low()
        self.high = self._get_high()

    def sample(self):
        #  return an array (1, space_length) 
        values = [spc.sample() for spc in self.spaces]
        return np.array(values).reshape(1, self.length)

    def contains(self, x):
        return all(spc.contains(part) for (spc,part) in zip(self.spaces,x))

    def discretize(self, num_discrete):
        #  use the same space length for all parts of the action space
        return [spc.discretize(num_discrete) for spc in self.spaces]

    def _get_shape(self):
        return (self.length,)

    def _get_low(self):
        lows = [spc.low for spc in self.spaces]
        return np.array(lows).reshape(-1)

    def _get_high(self):
        highs = [spc.high for spc in self.spaces]
        return np.array(highs).reshape(-1)
    
