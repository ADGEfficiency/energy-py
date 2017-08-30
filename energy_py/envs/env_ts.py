import numpy as np
import pandas as pd

from energy_py.envs.env_core import Base_Env
from energy_py.main.scripts.spaces import Continuous_Space, Discrete_Space

class Time_Series_Env(Base_Env):
    """
    The base environment class for time series environments.

    Most energy problems are time series problems - hence the need for a
    specific environment.
    """

    def __init__(self, episode_visualizer, lag, episode_length, episode_start, csv_path, verbose):
        self.lag = lag
        self.episode_start = episode_start
        self.episode_length = episode_length
        self.csv_path = csv_path

        super().__init__(episode_visualizer, verbose)

        self.raw_ts = self.load_ts_from_csv(self.csv_path)

    def ts_env_main(self):
        """
        The master function for the Time_Series_Env class.

        Envisioned that this will be run during the _reset of the child class.
        """

        #  creating the observation space list
        observation_space = self.make_env_obs_space(self.raw_ts)

        #  now grab the start & end indicies
        start, end = self.get_ts_row_idx(self.raw_ts.shape[0],
                                    self.episode_length,
                                    self.episode_start)

        #  use these to index the time series for this episode
        ep_ts = self.raw_ts.iloc[start:end]
        print('episode starting at  {}'.format(ep_ts.index[0]))

        #  now we make our state and observation dataframes
        observation_ts, state_ts = self.make_state_observation_ts(ep_ts, self.lag)

        return observation_space, observation_ts, state_ts

    def load_ts_from_csv(self, csv_path):
        """
        Loads a CSV
        """
        #  loading the raw time series data
        raw_ts = pd.read_csv(csv_path,
                             index_col=0)

        print('length of time series is '+str(raw_ts.shape[0]))
        print('cols of time series are '+str(raw_ts.columns))

        return raw_ts

    def get_ts_row_idx(self, ts_length, episode_length, episode_start):
        """

        """
        start = episode_start
        if episode_length == 'maximum':
            episode_length = ts_length - 1

        if episode_start == 'random':
            end_int_idx = ts_length - episode_length
            start = np.random.randint(0, end_int_idx)

        #  now we can set the end of the episode
        end = start + episode_length
        return start, end

    def make_env_obs_space(self, ts):
        """
        """
        observation_space = []

        for name, col in ts.iteritems():
            #  pull the label from the column name
            label = str(name[:2])

            if label == 'D_':
                obs_space = Discrete_Space(col.min(), col.max(), 1)

            elif label == 'C_':
                obs_space = Continuous_Space(col.min(), col.max())

            else:
                print('time series not labelled correctly')
                assert 1 == 0

            observation_space.append(obs_space)
        assert len(observation_space) == ts.shape[1]

        return observation_space

    def make_state_observation_ts(self, ts, lag):
        """
        Takes the processed time series and deals with the lags
        """

        #  offset = 0 -> state == observation
        if lag == 0:
            observation_ts = ts.iloc[:,:]
            state_ts = ts.iloc[:,:]

        #  offset = negative -> agent can only see past
        elif offset < 0:
            #  shift & cut observation
            observation_ts = ts.shift(lag).iloc[:-lag, :]
            #  we cut the state
            state_ts = ts.iloc[lag:, :]

        #  offset = positive -> agent can see the future
        elif offset > 0:
            #  shift & cut observation
            observation_ts = ts.shift(lag).iloc[lag:, :]
            #  cut the state
            state_ts = ts.iloc[lag:, :]

        assert observation_ts.shape == state_ts.shape
        if self.verbose > 0:
            print('observation time series shape is {}'.format(observation_ts.shape))

        return observation_ts, state_ts

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
