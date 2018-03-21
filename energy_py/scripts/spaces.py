"""
energy_py Space objects - inspired by the OpenAI gym Spaces.

Compatability with gym spaces is ideal as it allows energy_py agents
to be used with gym environments.

The energy_py GlobalSpace is the equivilant of the gym TupleSpace.

As GlobalSpaces are created by the environment, discrete representations of
the spaces that form the GlobalSpace are available on demand via

GlobalSpace.discretize(n_discr=10)
which sets .discrete_spaces

This leads to two issues
1 - appending on a new space without rediscretizing
2 - redisctretizig with different n_discr

Another option would be to never hold a discrete representation of the GlobalSpace.
Sampling would then be done by passing in the discretized space.

I'm trying to keep the API between all the spaces similar to chose to keep it
in the GlobalSpace object.
"""

import itertools

import numpy as np


class DiscreteSpace(object):
    """
    A single dimension discrete space
    - an on/off switch
    - a single button on a keyboard

    args
        num (int) the number of options across the discrete space
    """

    def __init__(self, num):
        self.low = 0
        self.high = num

    def sample(self):
        return np.random.randint(self.high + 1)

    def contains(self, x):
        return np.in1d(x, np.arange(0, self.high))

    def discretize(self, n_discr=[]):
        #  we don't use num discrete here
        return np.arange(0, self.high + 1)


class ContinuousSpace(object):
    """
    A single dimension continuous space
    - a car accelerator
    - load on a gas turbine
    - speed of a variable speed drive

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


class GlobalSpace(object):
    """
    A combination of simpler spaces

    This class is used directly as the action or observation space
    of environments and agents

    args
        spaces (list) a list of DiscreteSpace or ContinuousSpace
    """
    def __init__(self, spaces):
        assert type(spaces) is list
        self.spaces = [spc for spc in spaces]

    def sample(self):
        sample = [spc.sample() for spc in self.spaces]
        return np.array(sample).reshape(1, *self.shape)

    def sample_discrete(self):
        """
        Separate method for clarity when using the GlobalSpace

        This method requires that self.discrete_spaces has been set using
        the discretize() method.
        """
        idx = np.random.randint(0, len(self.discrete_spaces))
        return np.array(self.discrete_spaces[idx]).reshape(1, *self.shape)

    def contains(self, x):
        """
        Here we need to index the first element as energy_py observations or
        actions are always (batch_size, length)
        """
        assert x.ndim == 2
        return all(spc.contains(part) for (spc, part) in zip(self.spaces, x[0]))

    def discretize(self, n_discr):
        """
        Using the same n_discr across each dimension of the GlobalSpace

        args
            n_discr (int) number of discrete spaces in the action space
        """
        disc = [spc.discretize(n_discr) for spc in self.spaces]
        self.discrete_spaces = [list(a) for a in itertools.product(*disc)]
        return self.discrete_spaces

    def append(self, space):
        """
        args
            space (object) energy_py space object
        """
        self.spaces.append(space)

    @property
    def shape(self):
        return (len(self.spaces),)

    @property
    def high(self):
        return np.array([spc.high for spc in self.spaces])

    @property
    def low(self):
        return np.array([spc.low for spc in self.spaces])
