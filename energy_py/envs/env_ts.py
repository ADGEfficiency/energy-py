import logging

import numpy as np
import pandas as pd

from energy_py.envs import BaseEnv
from energy_py.scripts.spaces import ContinuousSpace, DiscreteSpace, GlobalSpace

class TimeSeriesEnv(BaseEnv):
    """
    The base environment class for time series environments

    Most energy problems are time series problems - hence the need for a
    class to give functionality
    """

    def __init__(self, 
                 episode_length,
                 episode_start,
                 state_path,
                 observation_path):

        self.episode_start = episode_start
        self.episode_length = episode_length
        self.state_path = state_path
        self.observation_path = observation_path

        #  load up the infomation from the csvs once
        #  we do this before we init the BaseEnv so we can reset in
        #  the BaseEnv class
        self.raw_state_ts = self.load_ts_from_csv(self.state_path)
        self.raw_observation_ts = self.load_ts_from_csv(self.observation_path)

        super().__init__()

    def get_state_obs(self):
        """
        The master function for the Time_Series_Env class

        Envisioned that this will be run during the _reset of the child class

        This is to allow different time periods to be sampled
        """

        #  creating the observation space list
        observation_space = self.make_env_obs_space(self.raw_state_ts)
        observation_space = GlobalSpace(spaces=observation_space)

        #  now grab the start & end indicies
        ts_length = min(self.raw_state_ts.shape[0],
                        self.raw_observation_ts.shape[0])
        start, end = self.get_ts_row_idx(ts_length,
                                         self.episode_length,
                                         self.episode_start)

        state_ts = self.raw_state_ts.iloc[start:end, :]
        observation_ts = self.raw_observation_ts.iloc[start:end, :]

        assert observation_ts.shape[0] == state_ts.shape[0]
        logging.info('Ep {} starting at {}'.format(self.episode,
                                                        state_ts.index[0]))
        logging.debug(state_ts.iloc[:,0].describe())

        return observation_space, observation_ts, state_ts

    def load_ts_from_csv(self, path):
        """
        Loads a CSV
        """
        #  loading the raw time series data
        raw_ts = pd.read_csv(path, index_col=0)
        return raw_ts

    def get_ts_row_idx(self, ts_length, episode_length, episode_start):
        """
        Gets the integer indicies for selecting the episode
        time period
        """
        start = episode_start
        if episode_length == 'maximum':
            episode_length = ts_length - 1
            self.episode_length = episode_length

        if episode_start == 'random':
            end_int_idx = ts_length - episode_length
            start = np.random.randint(0, end_int_idx)

        #  now we can set the end of the episode
        end = start + episode_length
        return start, end

    def make_env_obs_space(self, ts):
        """
        Creates the observation space list
        """
        observation_space = []

        for name, col in ts.iteritems():
            #  pull the label from the column name
            label = str(name[:2])

            if label == 'D_':
                obs_space = DiscreteSpace(col.min(), col.max(), 1)

            elif label == 'C_':
                obs_space = ContinuousSpace(col.min(), col.max())

            else:
                print('time series not labelled correctly')
                assert 1 == 0

            observation_space.append(obs_space)
        assert len(observation_space) == ts.shape[1]

        return observation_space

    def get_state(self, steps, append=[]):
        """
        Helper function to get a state.

        Also takes an optional argument to append onto the end of the array.
        This is so that environment specific info can be added onto the
        state or observation array.

        Repeated code with get_observation but I think having two functions
        is cleaner when using in the child class.
        """
        ts_info = np.array(self.state_ts.iloc[steps, :])
        ts_info = np.append(ts_info, append)
        return ts_info.reshape(-1)

    def get_observation(self, steps, append=[]):
        """
        Helper function to get a observation.

        Also takes an optional argument to append onto the end of the array.
        This is so that environment specific info can be added onto the
        state or observation array.
        """
        ts_info = np.array(self.state_ts.iloc[steps, :])

        ts_info = np.append(ts_info, np.array(append))

        return ts_info.reshape(-1)
