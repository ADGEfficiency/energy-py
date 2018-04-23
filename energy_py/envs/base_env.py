"""
The base class for energy_py environments
"""
import collections
import logging
import os
from random import randint

import numpy as np
import pandas as pd

import energy_py
from energy_py import ContinuousSpace, DiscreteSpace, load_csv

logger = logging.getLogger(__name__)


class BaseEnv(object):
    """
    The base environment class for time series environments

    args
        dataset_name (str) located in energy_py/experiments/datasets
        episode_length (int)
        episode_start (int) integer index of episode start
        episode_random (bool) whether to randomize the episode start position

    Most energy problems are time series problems
    The BaseEnv has functionality for working with time series data
    """
    def __init__(self,
                 dataset_name='test',
                 episode_start=0,
                 episode_random=False,
                 episode_length=48):

        logger.info('Initializing environment {}'.format(repr(self)))

        self.episode_start = int(episode_start)
        self.episode_random = bool(episode_random)
        self.episode_length = int(episode_length)

        #  load the time series info from csv
        self.state, self.observation = self.load_dataset(dataset_name)

    def _step(self, action): raise NotImplementedError

    def _reset(self): raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation

        returns 
            observation (np array) initial observation
        """
        logger.debug('Resetting environment')

        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        self.state_ep, observation_ep = self.get_episode()

        logger.debug('Episode start {} / 
                      Episode end {}'.format(self.state_ep.index[0]
                                             self.state_ep.index[-1]))
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

        User is responsible for resetting after episode end.

        The step function should progress in the following order:
        - action = a[1]
        - reward = r[1]
        - next_state = next_state[1]
        - update_info()
        - step += 1
        - self.state = next_state[1]

        step() returns the observation - not the state!
        """
        action = np.array(action).reshape(1, self.action_space.shape*)

        logger.debug('step {} action {}'.format(self.steps, action))

        return self._step(action)

    def update_info(self, **kwargs):
        """
        Helper function to update the self.info dictionary.
        """
        for name, data in kwargs.items():
            self.info[name].append(data)

        return self.info

    def load_dataset(self, dataset_name):
        """
        Loads time series data from csv

        args
            dataset_name (str)

        Datasets are located in energy_py/experiments/datasets
        A dataset consists of state.csv and observation.csv
        """
        logger.info('Using {} dataset'.format(dataset_name))
        logger.debug('Dataset path {}'.format(dataset_path))

        dataset_path = energy_py.get_dataset_path(dataset_name)

        try:
            state = load_csv(dataset_path, 'state.csv'))
            observation = load_csv(dataset_path, 'observation.csv'))

            assert state.shape[0] == observation.shape[0]

        except FileNotFoundError:
            raise FileNotFoundError('state.csv & observation.csv are missing /
                                    from {}'.format(dataset_path)))

        logger.debug('State start {} / 
                      State end {}'.format(self.state.index[0]
                                             self.state.index[-1]))

        logger.debug('State info {}'.format(self.state_info))
        logger.debug('Observation info {}'.format(self.observation_info))

        return state, observation

    def make_observation_space(self, spaces, space_labels):

        #  pull out the column names so we know what each variable is
        self.state_info = state.columns.tolist()
        self.observation_info = observation.columns.tolist()

        observation_space = make_obs_spaces(self.observation)

        observation_space.extend(spaces)
        observation_info.extend(space_labels)
        
        return GlobalSpace(observation_space)

    def get_episode(self):
        """
        Samples a single episode from the state and observation dataframes

        returns
            observation_ep (pd.DataFrame)
            state_ep (pd.DataFrame)
        """
        max_len = int(self.state.shape[0])
        start = self.episode_start

        if self.episode_length == 0:
            self.episode_length = max_len 

        if self.episode_random:
            end_idx = max_len - self.episode_length
            start = randint(0, end_idx)
        
        end = start + self.episode_length

        state_ep = self.state.iloc[start:end, :]
        observation_ep = self.observation.iloc[start:end, :]
        assert observation_ts.shape[0] == state_ts.shape[0]

        logging.debug('max_len {} start {} end {} random {}'.format(max_len,
                                                                    start,
                                                                    end,
                                                                    random))
        return state_ep, observation_ep

    def make_env_obs_space(self, obs_ts):
        """
        Creates the observation space list

        args
            obs_ts (pd.DataFrame)

        returns
            observation_space (list) contains energy_py Space objects
        """
        observation_space = []

        for name, col in obs_ts.iteritems():
            #  pull the label from the column name
            label = str(name[:2])

            if label == 'D_':
                obs_space = DiscreteSpace(col.max())

            elif label == 'C_':
                obs_space = ContinuousSpace(col.min(), col.max())

            else:
                raise ValueError('Time series columns mislabelled')

            observation_space.append(obs_space)

        assert len(observation_space) == obs_ts.shape[1]

        return observation_space

    def get_state(self, steps, append=None):
        """
        Helper function to get a state.

        Also takes an optional argument to append onto the end of the array.

        This is so that environment specific info can be added onto the
        state array.

        Repeated code with get_observation but I think having two functions
        is cleaner when using in the child class.

        args
            steps (int) used as a row index
            append (list) optional array to append onto the state

        returns
            state (np.array)
        """
        state = np.array(self.state_ep.iloc[steps, :])

        if append:
            state = np.append(state, append)

        return state.reshape(1, len(self.state_info))

    def get_observation(self, steps, append=None):
        """
        Helper function to get a observation.

        Also takes an optional argument to append onto the end of the array.

        This is so that environment specific info can be added onto the
        observation array.

        Repeated code with get_observation but I think having two functions
        is cleaner when using in the child class.

        args
            steps (int) used as a row index
            append (list) optional array to append onto the observation

        returns
            observation (np.array)
        """
        observation = np.array(self.observation_ep.iloc[steps, :])

        if append:
            observation = np.append(observation, np.array(append))

        return observation.reshape(1, self.observation_space.shape*)
