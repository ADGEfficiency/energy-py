import numpy as np
import pandas as pd

import gym
from gym import spaces
from gym.utils import seeding


class base_class(gym.Env):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _seed(self, seed=None):  # taken straight from cartpole
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.verbose = 0
        self.ts = self.load_data(self.episode_length)

        self.state_names = [d['Name'] for d in self.state_models]
        self.action_names = [var['Name']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.s_mins, self.s_maxs = self.state_mins_maxs()
        self.a_mins, self.a_maxs = self.asset_mins_maxs()
        self.mins = np.append(self.s_mins, self.a_mins)
        self.maxs = np.append(self.s_maxs, self.a_maxs)

        self.seed()
        self.steps = int(0)
        self.state = self.ts.iloc[0, 1:].values  # visible state
        self.info = []
        self.done = False
        [asset.reset() for asset in self.asset_models]
        self.last_actions = [var['Current']
                             for asset in self.asset_models
                             for var in asset.variables]

        self.observation_space = self.create_obs_space()
        self.action_space = self.create_action_space()
        return self.state

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Non-Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def load_data(self, episode_length):
        return self._load_data(episode_length)

    def create_action_space(self):
        return self._create_action_space()

    def create_obs_space(self):
        states, self.state_names = [], []
        for mdl in self.state_models:
            states.append([mdl['Min'], mdl['Max']])
            self.state_names.append(mdl['Name'])
        return spaces.MultiDiscrete(states)

    def state_mins_maxs(self):
        s_mins, s_maxs = np.array([]), np.array([])
        for mdl in self.state_models:
            s_mins = np.append(s_mins, mdl['Min'])
            s_maxs = np.append(s_maxs, mdl['Max'])
        return s_mins, s_maxs

    def asset_mins_maxs(self):
        a_mins, a_maxs = [], []
        for j, asset in enumerate(self.asset_models):
            for var in asset.variables:
                a_mins = np.append(a_mins, var['Min'])
                a_maxs = np.append(a_maxs, var['Max'])
        return a_mins, a_maxs

    def asset_states(self):
        for asset in self.asset_models:
            for var in asset.variables:
                print(var['Name'] + ' is ' + str(var['Current']))
        return self
