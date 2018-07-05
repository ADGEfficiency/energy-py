import collections
import logging

import numpy as np
import pandas as pd

import energy_py
from energy_py.common.spaces import ContinuousSpace, DiscreteSpace, GlobalSpace
from energy_py.common.utils import load_csv


logger = logging.getLogger(__name__)


class BaseEnv(object):
    """
    Generic time series environment

    args
        dataset (str) located in energy_py/experiments/datasets
        episode_sample (str) i.e. fixed, random
        episode_length (int)
    """
    def __init__(self,
                 dataset='example',
                 episode_sample='fixed',
                 episode_length=2016):

        logger.info('Initializing environment {}'.format(repr(self)))

        self.state_space = GlobalSpace('state').from_dataset(dataset)
        self.observation_space = GlobalSpace('observation').from_dataset(dataset)

        if episode_sample == 'random':
            self.sample_stragety = self.random_sample

        if episode_sample == 'fixed':
            self.sample_stragety = self.fixed_sample

        self.episode_length = min(
            int(episode_length),
            self.state_space.data.shape[0]
        )

    def reset(self):
        """
        Resets the state of the environment, returns an initial observation

        returns
            observation (np array) initial observation
        """
        logger.debug('Resetting environment')

        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        episode = self.sample_episode()
        self.state_space.episode = episode[0]
        self.observation_space.episode = episode[1]

        logger.debug(
            'Episode start {} Episode end {}'.format(
                self.state_space.episode.index[0],
                self.state_space.episode.index[-1])
        )

        return self._reset()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        args
            action (object) an action provided by the environment
            episode (int) the current episode number

        returns
            observation (np array) agent's observation of the environment
            reward (np.float)
            done (boolean)
            info (dict) auxiliary information
        """
        action = np.array(action).reshape(1, *self.action_space.shape)
        assert self.action_space.contains(action)

        logger.debug('step {} action {}'.format(self.steps, action))

        return self._step(action)

    def sample_episode(self):
        """ Samples a single episode """
        start, end = self.sample_stragety()
        print('Sampling episode start {} end {}'.format(start, end))

        state_ep = self.state_space.sample_episode(start, end)
        obs_ep = self.observation_space.sample_episode(start, end)

        assert state_ep.shape[0] == obs_ep.shape[0]
        return state_ep, obs_ep

    def random_sample(self):
        start = np.random.randint(
            low=0,
            high=self.state_space.data.shape[0] - self.episode_length
        )
        return start, start + self.episode_length

    def fixed_sample(self):
        if self.episode_length == 0:
            ep_len = self.state_space.data.shape[0],
        else:
            ep_len = self.episode_length

        start = self.state_space.data.shape[0] - self.episode_length

        return start, start + ep_len

    def get_state_variable(self, variable_name):
        return self.state[self.state_space.info.index(variable_name)][0]

    def update_info(self, **kwargs):
        for name, data in kwargs.items():
            self.info[name].append(data)

        return self.info
