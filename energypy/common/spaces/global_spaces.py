""" state, observation and action spaces """

from collections import namedtuple, OrderedDict
from itertools import product

import numpy as np

import energypy as ep
from energypy.common.spaces import DiscreteSpace, ContinuousSpace


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
            self[p.name] = space_register[p.type](p.name, p.low, p.high, data=p.data) 

        self.num_samples = self.set_num_samples()
        return self

    def set_num_samples(self):
        num_samples = []
        for space in self.values():
            if space.data is not None:
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
        append = {name: data}
        """
        data = []
        for name, space in self.items():

            if name in append.keys():
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
        if how == 'full':
            return 0, self.num_samples

        elif how == 'random':

            if self.num_samples == episode_length:
                start = 0
            else:
                start = np.random.randint(
                    low=0, high=self.num_samples - episode_length
                )
            return start, start + episode_length

        elif how == 'fixed':
            return 0, episode_length

        else:
            raise ValueError


class ActionSpace(Space):
    def __init__(self, name='action'):
        super().__init__(name=name)

    def discretize(self, num_discrete):
        #  get the discretized elements for each dim of global space
        discrete = [s.discretize(num_discrete) for s in self]

        #  get all combinations of each space (this is curse of dimensionality)
        discrete = [comb for comb in product(*discrete)]

        return np.array(discrete).reshape(-1, *self.shape)


class ObservationSpace():
    def __init__(self):
        pass


PrimCfg = namedtuple(
    'primitive', ['name', 'low', 'high', 'type', 'data']
)

space_register = {
    'discrete': DiscreteSpace,
    'continuous': ContinuousSpace
}

if __name__ == '__main__':

    prices = 10 * np.random.rand(10)
    batt = ep.make_env('battery', prices=prices, episode_length=20)

    #  episode sample happens during reset
    obs = batt.reset()
    done = False

    while not done:
        obs, r, done, info = batt.step(batt.action_space.sample())
