import logging
import itertools

import pandas as pd
import numpy as np

import energy_py
from energy_py.common.spaces.discrete import DiscreteSpace
from energy_py.common.spaces.continuous import ContinuousSpace


logger = logging.getLogger(__name__)


class GlobalSpace(object):
    """
    A combination of simpler spaces

    args
        name (str)
        dataset (str)
    """

    def __init__(
            self,
            name,
    ):
        self.name = name

    def from_dataset(self, dataset='example'):
        self.load_dataset(dataset)
        return self

    def from_spaces(self, spaces, space_labels):
        if not isinstance(spaces, list):
            spaces = [spaces]

        self.spaces = spaces
        self.info = space_labels
        return self

    @property
    def shape(self):
        return (len(self.spaces), )

    def extend(self, space, label):
        if not isinstance(space, list):
            space = [space]

        self.spaces.extend(space)

        if not isinstance(label, list):
            label = [label]

        self.info.extend(label)

    def contains(self, x):
        return all(
            spc.contains(part) for (spc, part) in zip(self.spaces, x[0])
        )

    def sample(self):
        return np.array([spc.sample() for spc in self.spaces]).reshape(1, *self.shape)

    def sample_discrete(self):
        if not hasattr(self, 'discrete_spaces'):
            raise ValueError('space is not discrete - call self.discretize')

        return np.array(
            self.discrete_spaces[np.random.randint(0, len(self.discrete_spaces))]
        ).reshape(1, *self.shape)

    def discretize(self, num_discrete):
        discrete_spaces = [spc.discretize(num_discrete) for spc in self.spaces]
        discrete_spaces = [a for a in itertools.product(*discrete_spaces)]

        return np.array(
            discrete_spaces).reshape(len(discrete_spaces), *self.shape)

    def load_dataset(self, dataset):
        self.data = energy_py.load_dataset(dataset, self.name)

        self.info = self.data.columns.tolist()
        logger.debug('{} info {}'.format(self.name, self.info))

        self.spaces = self.generate_spaces()
        assert len(self.spaces) == self.data.shape[1]

        return self.data

    def generate_spaces(self):
        spaces = []

        for name, col in self.data.iteritems():
            label = str(name[:2])

            if label == 'D_':
                space = DiscreteSpace(col.max())

            elif label == 'C_':
                space = ContinuousSpace(col.min(), col.max())

            else:
                raise ValueError('Time series columns mislabelled')

            spaces.append(space)

        return spaces

    def sample_episode(self, start, end):
        self.episode = self.data.iloc[start: end, :]
        return self.episode

    def __call__(self, steps, append=None):
        sample = np.array(self.episode.iloc[steps, :])

        if append:
            sample = np.append(sample, append)

        return sample.reshape(1, *self.shape) 
