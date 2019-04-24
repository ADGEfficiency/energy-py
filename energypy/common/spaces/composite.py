""" state, observation and action spaces """

from collections import namedtuple, OrderedDict
from io import BytesIO
from itertools import product
from os.path import join
import pkg_resources

import numpy as np
import pandas as pd

import energypy as ep
from energypy.common.spaces import DiscreteSpace, ContinuousSpace


#  used in envs
PrimitiveConfig = namedtuple(
    'primitive', ['name', 'low', 'high', 'type', 'data']
)


primitive_register = {
    'discrete': DiscreteSpace,
    'continuous': ContinuousSpace
}


class Space(OrderedDict):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self._shape = None

    def __repr__(self):
        return('<{} space {}>'.format(self.name, self.shape))

    @property
    def shape(self):
        return self._shape

    @shape.getter
    def shape(self):
        return (len(self.keys()), )

    @property
    def low(self):
        return self._low

    @low.getter
    def low(self):
        return np.array([spc.low for spc in self.values()]).reshape(*self.shape)

    @property
    def high(self):
        return self._high

    @high.getter
    def high(self):
        return np.array([spc.high for spc in self.values()]).reshape(*self.shape)

    def sample(self):
        return np.array([spc.sample() for spc in self.values()]).reshape(1, *self.shape)

    def contains(self, x):
        return all(spc.contains(part) for (spc, part) in zip(self.values(), x[0]))

    def from_primitives(self, *primitives):
        for p in primitives:
            self[p.name] = primitive_register[p.type](p.name, p.low, p.high, data=p.data) 
        self.num_samples = self.set_num_samples()
        return self

    def append(self, primitive):
        p = primitive 
        self[p.name] = primitive_register[p.type](p.name, p.low, p.high, data=p.data) 
        self.num_samples = self.set_num_samples()
        return self

    def set_num_samples(self):
        num_samples = []
        for name, space in self.items():

            if isinstance(space.data, str):
                assert space.data == 'append'

            else:
                num_samples.append(np.array(space.data).shape[0])

            if num_samples:
                assert max(num_samples) == min(num_samples)
                return max(num_samples)
            else:
                return None


class StateSpace(Space):
    def __init__(self, name='state'):
        super().__init__(name=name)

    def __call__(self, steps, offset, append=None):
        """
        steps = num steps through episode
        start = offset for start of episode
        end = offset for end of episode
        append = {name: data}, data from env appended to state / obs
        """
        data = []
        for name, space in self.items():

            if space.data == 'append':
            # if isinstance(space.data, str):
                assert space.data == 'append'
                data.append(append[name])

            elif space.data is not None:
                data.append(space(steps, offset))

            else:
                raise ValueError

        return np.array(data).reshape(1, *self.shape)

    def sample_episode(
            self,
            how='full',
            episode_length=None
    ):
        if episode_length:
            episode_length = min(episode_length, self.num_samples)

        if how == 'full':
            return 0, self.num_samples

        elif how == 'random':

            if self.num_samples == episode_length:
                return 0, episode_length
            else:
                start = np.random.randint(
                    low=0, high=self.num_samples - episode_length
                )
            return start, start + episode_length

        elif how == 'fixed':
            return 0, episode_length

        else:
            raise ValueError

    def from_dataset(self, dataset):

        data = self.load_dataset(dataset)

        for col in data.columns:
            d = np.array(data.loc[:, col]).reshape(-1)

            #  TODO doing all as continuous spaces here!
            self[col] = primitive_register['continuous'](
                col, np.min(d), np.max(d), d)

        self.num_samples = self.set_num_samples()

        return self

    def load_dataset(self, dataset):
        """ load example dataset or load from user supplied path """
        if dataset == 'example':
            data = pkg_resources.resource_string(
                'energypy',
                'examples/{}.csv'.format(self.name)
            )
            return pd.read_csv(BytesIO(data), index_col=0, parse_dates=True)

        else:
            return pd.read_csv(join(dataset, self.name + '.csv'), index_col=0, parse_dates=True)


class ActionSpace(Space):
    def __init__(self, name='action'):
        super().__init__(name=name)

    def discretize(self, num_discrete):
        #  get the discretized elements for each dim of global space
        discrete = [spc.discretize(num_discrete) for spc in self.values()]

        #  get all combinations of each space (this is curse of dimensionality)
        discrete = [comb for comb in product(*discrete)]

        return np.array(discrete).reshape(-1, *self.shape)


class ObservationSpace():
    def __init__(self):
        raise NotImplementedError
