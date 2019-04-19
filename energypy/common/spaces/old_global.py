import itertools

import numpy as np
from sklearn.preprocessing import StandardScaler

import energypy
from energypy.common.spaces import DiscreteSpace, ContinuousSpace


class GlobalSpace(object):
    """
    A combination of simpler spaces

    TODO - raise exception if try to call .discrete_spaces
    before running discretize()

    args
        name (str)
        dataset (str)
    """

    def __init__(
            self,
            name,
    ):
        self.name = name
        self._shape = None

    def __repr__(self):
        return('<{} space {}>'.format(self.name, self.shape))

    def __call__(self, steps, append=None):
        sample = np.array(self.episode.iloc[steps, :])

        #  needed because bool(np.array(0)) is falsy
        if isinstance(append, np.ndarray):
            sample = np.append(sample, append)

        return sample.reshape(1, *self.shape)

    @property
    def shape(self):
        return self._shape

    @shape.getter
    def shape(self):
        return (len(self.spaces), )

    @property
    def low(self):
        return self._low

    @low.getter
    def low(self):
        return np.array([spc.low for spc in self.spaces]).reshape(*self.shape)

    @property
    def high(self):
        return self._low

    @high.getter
    def high(self):
        return np.array([spc.high for spc in self.spaces]).reshape(*self.shape)

    @property
    def num_discrete_spaces(self):
        return self._discrete_spaces

    @num_discrete_spaces.getter
    def num_discrete_spaces(self):
        try:
            len(self.discrete_spaces)
        except AttributeError:
            self.discrete_spaces = self.discretize()

        return len(self.discrete_spaces)

    def sample_discrete(self):
        try:
            len(self.discrete_spaces)
        except AttributeError:
            self.discrete_spaces = self.discretize()

        return np.array(
            self.discrete_spaces[np.random.randint(0, len(self.discrete_spaces))]
        ).reshape(1, *self.shape)

    def discretize(self, num_discrete=None):
        self.discrete_spaces = np.array([
            a for a in itertools.product(
                *[spc.discretize(num_discrete) for spc in self.spaces])
        ]).reshape(-1, *self.shape)

        return self.discrete_spaces

    def from_dataset(self, dataset='example'):
        data = energypy.load_dataset(dataset, self.name)

        if self.name == 'observation':
            data.loc[:, :] = StandardScaler().fit_transform(data.values)

        self.data = data

        self.info = self.data.columns.tolist()

        self.spaces = self.generate_spaces()
        assert len(self.spaces) == self.data.shape[1]

        return self

    def from_spaces(self, spaces, labels):
        if not isinstance(spaces, list):
            spaces = [spaces]

        if not isinstance(labels, list):
            labels = [labels]

        self.spaces = spaces
        self.info = labels

        return self

    def extend(self, space, label):
        if not isinstance(space, list):
            space = [space]

        self.spaces.extend(space)

        if not isinstance(label, list):
            label = [label]

        self.info.extend(label)

        assert len(self.spaces) == len(self.info)

    def contains(self, x):
        return all(
            spc.contains(part) for (spc, part) in zip(self.spaces, x[0])
        )

    def sample(self):
        return np.array([spc.sample() for spc in self.spaces]).reshape(1, *self.shape)

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

    def no_op(self):
        raise NotImplementedError(
            'implement this in the environment child class (ie battery, flex etc'
        )
