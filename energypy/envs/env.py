import collections
import json
import random
import tensorflow as tf

import numpy as np

import energypy as ep


class BaseEnv(object):
    """ time series environment """
    def __init__(
            self,
            seed=None,
            dataset='example',
    ):
        if seed:
            self.seed(seed)

    def seed(self, seed):
        """ sets random seeds """
        seed = int(seed)
        random.seed(seed)
        tf.set_random_seed(seed)
        np.random.seed(seed)

    def reset(self):
        """ resets the environment, returns an initial observation """
        self.steps = 0
        self.done = False

        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        if not hasattr(self, 'episode_logger'):
            self.episode_logger = ep.common.make_new_logger('episode')

        return self._reset()

    def step(self, action, log=True):
        """ run one timestep of the environment's dynamics """
        if not hasattr(self, 'state'):
            raise ValueError(
                'You need to reset the environment before calling step()')

        action = np.array(action).reshape(1, *self.action_space.shape)
        assert self.action_space.contains(action)

        #  child class specific transition dynamics
        transition = self._step(action)

        for k, v in transition.items():
            transition[k] = np.array(v).tolist()
            self.info[k].append(v)

        #  TODO
        if log:
            #  episode logger is set during experiment
            self.episode_logger.debug(json.dumps(transition))

        return self.observation, self.reward, self.done, self.info

    def get_state_variable(self, variable_name):
        """ get single element of the current state """
        idx = list(self.state_space.keys()).index(variable_name)
        return self.state[0][idx]

    def update_info(self, **kwargs):
        for name, data in kwargs.items():
            self.info[name].append(data)
        return self.info
