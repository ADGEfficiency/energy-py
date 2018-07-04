import logging
import itertools

import numpy as np

from energy_py.common.spaces.discrete import DiscreteSpace
from energy_py.common.spaces.continuous import ContinuousSpace

logger = logger.getLogger(__name__)


def test_index_length(df, freq):
    test_idx = pd.DatetimeIndex(
        start=df.index[0],
        end=df.index[-1],
        freq=freq
    )
    assert test_idx.shape[0] == state.shape[0]


class GlobalSpace(object):
    """
    A combination of simpler spaces

    args
        name (str)
        dataset (str)
    """

    def __init__(
            self
            name,
    ):
        self.name = name

    def from_dataset(self, dataset='example'):
        self.data = load_dataset(dataset)

    def from_spaces(self, spaces, space_labels):
        self.spaces = spaces
        self.info = space_labels

    @property
    def shape(self):
        return (len(self.spaces), )

    def extend(self, space):
        self.spaces.extend(space)

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
        dataset_path = energy_py.get_dataset_path(dataset)
        logger.info('Dataset path {}'.format(dataset_path))

        #  TODO the check on length of state and obser at higher level!
        self.data = load_csv(dataset_path. '{}.csv'.format(name))

        test_index_length(self.data, '5min')

        except FileNotFoundError:
            raise FileNotFoundError(
                '{}.csv is missing from {}'.format(self.name, dataset_path)
            )

        logger.debug('loaded {} - start {} end {}'.format(
                          self.name, state.index[0], state.index[-1])
                     )

        self.info = self.data.columns.tolist()
        logger.debug('{} info {}'.format(self.name, self.info))

        self.spaces = generate_spaces()

        assert len(self.spaces) == self.data.shape[1]

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

            spaces.append(obs_space)

        return spaces

    def sample_episode(self, start, end):
        return self.data.iloc[start: end, :]

    def __getitem__(self, steps, append=None):
        sample = np.array(self.episode.iloc[steps, :])

        if append:
            sample = np.append(sample, append)

        return sample.reshape(1, *self.shape) 
