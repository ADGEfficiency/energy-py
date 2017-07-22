import collections
import random

import gym
import numpy as np
import pandas as pd

import environments.base_env
import environments.library

#  state = [demand (MW), price (£/MWh)]
#  action = [flex (MW)]
#  reward = (demand - flex) * price

class env(environments.base_env.base_class):

    def __init__(self,
                 episode_length,
                 lag,
                 random_ts,
                 verbose,
                 FLEX_max_capacity=10,  #  MW
                 FLEX_memory=4):    #  number of time steps

        self.episode_length = episode_length
        self.lag = lag
        self.random_ts = random_ts
        self.verbose = verbose

        self.state_models = []
        self.asset_models = [environments.library.flex(max_capacity=5,
                                                       memory=5,
                                                       name='1 ')]
        self.actual_state, self.visible_state = self.load_data(self.episode_length, self.lag, self.random_ts)

        self.state = self.reset()


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _step(self, action):
        actual_state = self.actual_state.iloc[self.steps, :]
        site_demand = actual_state.loc['site_demand']
        price = actual_state.loc['electricity_price']

        #  calculating the amount of flexible capacity used
        FLEX_capacity = [asset.capacity for asset in self.asset_models]
        FLEX_used = min(action, FLEX_capacity)
        import_power = actual_state.loc['site_demand'] - sum(FLEX_used)  # negative means exporting
        print(FLEX_capacity)
        reward = import_power * price
        # updating our assets
        for i, asset in enumerate(self.asset_models):
            asset.update(FLEX_used[i])

        self.steps += int(1)
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True
            a = self.
        next_state = self.visible_state.iloc[self.steps, 1:] # visible state
        self.state = next_state

        self.action_space = self.create_action_space()

        return next_state, reward, self.done, self.info

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Non-Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _load_data(self, episode_length, lag, random_ts):
        ts = pd.read_csv('environments/csvs/FLEX_time_series.csv',
                         index_col=0,
                         parse_dates=True)
        if lag < 0:
            actual_state = ts.iloc[:lag, :]
            visible_state = ts.shift(lag).iloc[:lag, :]

        elif lag == 0:
            actual_state = ts.iloc[:, :]
            visible_state = ts.iloc[:, :]

        elif lag > 0:
            actual_state = ts.iloc[lag:, :]
            visible_state = ts.shift(lag).iloc[lag:, :]

        assert actual_state.shape == visible_state.shape

        for col in actual_state.columns:
            self.state_models.append({'Name':col,
                                      'Min':actual_state.loc[:, col].min(),
                                      'Max':actual_state.loc[:, col].max()})

        return visible_state, actual_state

    def _create_action_space(self):
        #  TODO repeated code
        action_space = []
        for j, asset in enumerate(self.asset_models):
            radius = asset.variables[0]['Radius']
            space = gym.spaces.Box(low=0,
                                   high=radius,
                                   shape=(1))
            action_space.append(space)
        return action_space


    def _get_test_state_actions(self):
        self.Q_test = pd.read_csv('environments/csvs/FLEX_Q_test.csv',
                                  index_col=0,
                                  parse_dates=True)
        print(self.Q_test)
        print('q test shape {}'.format(self.Q_test.shape))
        return self.Q_test
