import collections
import logging
import os

import numpy as np
import pandas as pd

from energy_py.scripts.spaces import ContinuousSpace, DiscreteSpace

logger = logging.getLogger(__name__)


class BaseEnv(object):
    """
    The base environment class for time series environments

    Most energy problems are time series problems - hence the need for a
    class to give functionality.  Likely that this could be merged with the
    BaseEnv class

    args
        data_path (str) location of state.csv, observation.csv
        episode_length (int)
        episode_start (int) integer index of episode start
        episode_random (bool) whether to randomize the episode start position
    """

    def __init__(self,
                 data_path,
                 episode_length,
                 episode_start,
                 episode_random):

        self.episode_length = int(episode_length)
        self.episode_start = int(episode_start)
        self.episode_random = bool(episode_random)

        #  loads time series infomation from disk
        #  creates the state_info and observation_info lists
        self.raw_state_ts, self.raw_observation_ts = self.load_ts(data_path)

        #  hack to allow max length
        if self.episode_length == 0:
            self.episode_length = int(self.raw_observation_ts.shape[0])

        logger.info('Initializing environment {}'.format(repr(self)))

    # Override in subclasses
    def _step(self, action): raise NotImplementedError

    def _reset(self): raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (np array): the initial observation
        """
        logger.debug('Reset environment')

        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        return self._reset()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        User is responsible for resetting after episode end.

        The step function should progress in the following order:
        - action = a[1]
        - reward = r[1]
        - next_state = next_state[1]
        - update_info()
        - step += 1
        - self.state = next_state[1]

        step() returns the observation - not the state!

        args
            action (object) an action provided by the environment
            episode (int) the current episode number

        returns:
            observation (np array) agent's observation of the environment
            reward (np.float)
            done (boolean)
            info (dict) auxiliary information
        """
        action = np.array(action)
        action = action.reshape(1, self.action_space.shape[0])

        logger.debug('step {} action {}'.format(self.steps,
                                                action))
        return self._step(action)

    def update_info(self, **kwargs):
        """
        Helper function to update the self.info dictionary.
        """
        for name, data in kwargs.items():
            self.info[name].append(data)

        return self.info

    def load_ts(self, data_path):
        """
        Loads the state and observation from disk.

        args
            data_path (str) location of state.csv, observation.csv

        returns
            state (pd.DataFrame)
            observation (pd.DataFrame)
        """
        #  paths to load state & observation
        state_path = os.path.join(data_path, 'state.csv')
        obs_path = os.path.join(data_path, 'observation.csv')

        try:
            #  load from disk
            state = pd.read_csv(state_path, index_col=0)
            observation = pd.read_csv(obs_path, index_col=0)

        except FileNotFoundError:
            raise FileNotFoundError(('state.csv & observation.csv'
                                     'are missing from {}'.format(data_path)))

        #  grab the column name so we can index state & obs arrays
        self.state_info = state.columns.tolist()
        self.observation_info = observation.columns.tolist()
        logger.info('state info is {}'.format(self.state_info))
        logger.info('observation info is {}'.format(self.observation_info))

        assert state.shape[0] == observation.shape[0]

        return state, observation

    def get_state_obs(self):
        """
        Indexes the raw state & observation dataframes into smaller
        state and observation dataframes.

        returns
            observation_space (object) energy_py GlobalSpace
            observation_ts (pd.DataFrame)
            state_ts (pd.DataFrame)
        """
        start, end = self.get_ts_row_idx()

        state_ts = self.raw_state_ts.iloc[start:end, :]
        observation_ts = self.raw_observation_ts.iloc[start:end, :]

        #  creating the observation space list
        observation_space = self.make_env_obs_space(observation_ts)

        assert observation_ts.shape[0] == state_ts.shape[0]

        return observation_space, observation_ts, state_ts

    def get_ts_row_idx(self):
        """
        Sets the start and end integer indicies for episodes.

        returns
            self.episode_start (int)
            self.episode_end (int)
        """
        ts_length = self.raw_observation_ts.shape[0]

        if self.episode_random:
            end_int_idx = ts_length - self.episode_length
            self.episode_start = np.random.randint(0, end_int_idx)

        #  now we can set the end of the episode
        self.episode_end = self.episode_start + self.episode_length

        return self.episode_start, self.episode_end

    def make_env_obs_space(self, obs_ts):
        """
        Creates the observation space list.

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
        state = np.array(self.state_ts.iloc[steps, :])

        if append:
            state = np.append(state, append)

        state = state.reshape(1, -1)
        assert state.shape[1] == len(self.state_info)

        return state

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
        observation = np.array(self.observation_ts.iloc[steps, :])

        if append:
            observation = np.append(observation, np.array(append))

        observation = observation.reshape(1, -1)
        assert observation.shape[1] == len(self.observation_info)

        return observation
