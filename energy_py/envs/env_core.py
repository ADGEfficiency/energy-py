import collections
import os

import numpy as np
import pandas as pd


class Base_Env(object):
    """
    the energy_py base environment class
    inspired by the gym.Env class

    The methods of this class are:
        step
        reset

    To implement an environment:
    1 - override the following methods in your child:
        _step
        _reset

    2 - set the following attributes
        action_space
        observation_space
        reward_range (defaults to -inf, +inf)
    """

    def __init__(self, episode_visualizer, episode_length, episode_start, verbose):
        self.episode_visualizer_obj = episode_visualizer
        self.episode_length = episode_length
        self.episode_start = episode_start
        self.verbose = verbose

        self.info       = collections.defaultdict(list)
        self.episode    = None
        return None

    # Override in ALL subclasses
    def _step(self, action): raise NotImplementedError
    def _reset(self): raise NotImplementedError
    def _output_results(self): raise NotImplementedError

    #  Set these in ALL subclasses
    action_space = None       #  list of length num_actions
    observation_space = None  #  list of length obs_dim
    reward_space = None       #  single space object

    def load_state(self, csv_path, lag):
        """
        loads state infomation from a csv

        length = 2016 defaults to one week at 5 minuute time frequency
        """

        #  loading time series data
        ts = pd.read_csv(csv_path,
                         index_col=0,
                         parse_dates=True)
        ts_length = ts.shape[0]

        #  changing the episode length if we want to run the whole time series
        if self.episode_length == 'maximum':
            self.episode_length = ts_length - 1

        if self.episode_start == 'random':
            #  indexing the time series for a random time period
            start = np.random.randint(0, ts_length - self.episode_length)

        else:
            #  starting at user defined input
            start = self.episode_start
       
        #  now we can send the end of the episode
        end = start + self.episode_length

        #  indexing the dataframe
        ts = ts.iloc[start:end+1]

        #  now a three block if statement to deal with the lag
        #  1 - lag is 0
        #  2 - lag is negative
        #  3 - lag is positive

        #  if no lag then state = observation
        if lag == 0:
            observation_ts = ts.iloc[:, :]
            state_ts = ts.iloc[:, :]

        #  a negative lag means the agent can only see the past
        elif lag < 0:
            #  we shift & cut the observation
            observation_ts = ts.shift(lag).iloc[:-lag, :]
            #  we cut the state
            state_ts = ts.iloc[:-lag, :]

        #  a positive lag means the agent can see the future
        elif lag > 0:
            #  we shift & cut the observation
            observation_ts = ts.shift(lag).iloc[lag:, :]
            #  we cut the state
            state_ts = ts.iloc[lag:, :]

        #  checking our two ts are the same shape
        assert observation_ts.shape == state_ts.shape
        if self.verbose > 0:
            print('observation time series shape is {}'.format(observation_ts.shape))
            print('observation time series columns are {}'.format(observation_ts.columns))

        return observation_ts, state_ts

    def step(self, action, episode):
        """
        Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling reset().

        Accepts an action and returns a tuple (observation, reward, done, info).

        The step function should progress in the following order:
        - action = a[1]
        - reward = r[1]
        - next_state = next_state[1]
        - update_info()
        - step += 1
        - self.state = next_state[1]

        step() returns the observation - not the state!
        This is to allow the state to remain hidden (if desired by the modeller).

        Args:
            action  (object): an action provided by the environment
            episode (int): the current episode number

        Returns:
            observation (np array): agent's observation of the current environment
            reward (np float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        #  update the current episode number
        self.episode = episode

        return self._step(action)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (np array): the initial observation
        """
        if self.verbose > 0:
            print('Reset environment')
        self.episode_visualizer = None
        self.episode = None
        return self._reset()


    def output_results(self):
        """
        Initializes the visalizer object.
        """
        #  initalize the visualizer object with the current environment info
        self.episode_visualizer = self.episode_visualizer_obj(env_info=self.info, state_ts=self.state_ts, episode=self.episode)
        #  runs the main visualizer method
        _ = self.episode_visualizer.output_results()
        return self._output_results()
