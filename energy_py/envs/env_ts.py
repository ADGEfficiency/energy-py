import logging
import os

import numpy as np
import pandas as pd

from energy_py.envs import BaseEnv
from energy_py.scripts.spaces import ContinuousSpace, DiscreteSpace

logger = logging.getLogger(__name__)


def make_observation(path, horizion=5):
    """
    Creates the state.csv and observation.csv.

    Currently only supports giving the agent a forecast.

    args
        horizion (int) number of steps for the observation forecast
    """
    print('creating new state.csv and observation.csv')
    #  load the raw state infomation
    raw_state_path = os.path.join(path, 'raw_state.csv')
    raw_state = pd.read_csv(raw_state_path, index_col=0, parse_dates=True)

    #  create the observation, which is a perfect forecast
    observations = [raw_state.shift(-i) for i in range(horizion)]
    observation = pd.concat(observations, axis=1).dropna()

    #  we dropped na's we now need to realign
    state, observation = raw_state.align(observation, axis=0, join='inner')

    #  add a counter for agent to learn from
    observation['D_counter'] = np.arange(observation.shape[0])

    #  add some datetime features
    observation.index = state.index
    observation['D_hour'] = observation.index.hour

    state.to_csv(os.path.join(path, 'state.csv'))
    observation.to_csv(os.path.join(path, 'observation.csv'))

    return state, observation


class TimeSeriesEnv(BaseEnv):
    """
    The base environment class for time series environments

    Most energy problems are time series problems - hence the need for a
    class to give functionality

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

        #  load up the infomation from the csvs once
        #  we do this before we init the BaseEnv so we can reset in
        #  the BaseEnv class
        self.raw_state_ts, self.raw_observation_ts = self.load_ts(data_path)

        #  hack to allow max length 
        if self.episode_length == 0:
            self.episode_length = int(self.raw_observation_ts.shape[0])

        #  initialize BaseEnv parent class
        super().__init__()

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
            #  create observation from scratch
            state, observation = make_observation(data_path)

        #  grab the column name so we can index state & obs arrays
        self.state_info = state.columns.tolist()
        self.observation_info = observation.columns.tolist()

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
        logger.info('Ep starting at {}'.format(state_ts.index[0]))

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
            ts_info (np.array)

        """
        ts_info = np.array(self.state_ts.iloc[steps, :])
        ts_info = np.append(ts_info, append)

        return ts_info.reshape(1, -1)

    def get_observation(self, steps, append=[]):
        """
        Helper function to get a observation.

        Also takes an optional argument to append onto the end of the array.
        This is so that environment specific info can be added onto the
        observation array.

        args
            steps (int) used as a row index
            append (list) optional array to append onto the observation

        returns
            ts_info (np.array)
        """
        ts_info = np.array(self.observation_ts.iloc[steps, :])
        ts_info = np.append(ts_info, np.array(append))

        return ts_info.reshape(1, -1)
