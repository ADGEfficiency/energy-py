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
    A single dimension discrete space.
    - an on/off switch
    - a single button on a keyboard

    args
        num (int): the number of options across the discrete space
    """

    def __init__(self, num):
        self.num = num

    def sample(self):
        return np.random.choice(np.arange(0, self.num))

    def contains(self, x):
        return np.in1d(x, np.arange(0, self.num))

    def discretize(self, n_discr=[]):
        #  note that we don't use num discrete here
        return np.arange(0, self.num)


class ContinuousSpace(object):
    """
    A single dimension continuous space.
    - a car accelerator
    - load on a gas turbine
    - speed of a variable speed drive

    args
        low  (float) : an array with the minimum bound for each
        high (float) : an array with the maximum bound for each
    """

    def __init__(self, low, high):
        self.low  = float(low)
        self.high = float(high)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        return (x >= self.low) and (x <= self.high)

    def discretize(self, n_discr):
        #  we are using n_discr here
        return np.linspace(self.low, self.high, n_discr)


class GlobalSpace(object):
    """
    A combination of multiple ContinuousSpace or DiscreteSpace spaces

    All energy_py environments use this as the observation.space
    or action.space object.

    This is what any agent will be dealing with.

    Similar to the OpenAI gym TupleSpace object.

    Potential issue if user adds space then doesn't rediscretize
    Also issue if user changes n_discr

    Will add ability for a differ n_discr for each space eventually (easy)

    args
        spaces (list): a list of DiscreteSpace or ContinuousSpace
    """
    def __init__(self, spaces):
        #  our space is a list of the simpler spaces
        assert type(spaces) is list
        self.spaces = [spc for spc in spaces]

    def sample(self):
        sample = [spc.sample() for spc in self.spaces]
        sample = np.array(sample).reshape(1, self.shape[0])
        return sample

    def sample_discrete(self):
        """
        Chose to have a separate method for clarity when using the GlobalSpace
        Can probably do in a single line...
        """
        idx = np.random.randint(0, self.discrete_spaces.shape[0])
        sample = np.array(self.discrete_spaces[idx]).reshape(1, self.shape[0])
        return sample

    def contains(self, x):
        """
        Here we need to index the first element as energy_py observations or
        actions are always (num, length)
        """
        return all(spc.contains(part) for (spc, part) in zip(self.spaces, x[0]))

    def discretize(self, n_discr):
        #  using the same n_discr across the spaces
        disc = [spc.discretize(n_discr) for spc in self.spaces]
        self.discrete_spaces = np.array([a for a in itertools.product(*disc)])
        return self.discrete_spaces

    def append(self, space):
        self.spaces.append(space)

    @property
    def shape(self):
        return (len(self.spaces),)
