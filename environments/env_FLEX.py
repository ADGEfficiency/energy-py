import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

import environments.base_env
import environments.library


class env(environments.base_env.base_class):

    def __init__(self, episode_length, lag, random_ts, verbose, DSR_limit=10):
        self.episode_length = episode_length
        self.lag = lag
        self.random_ts = random_ts
        self.verbose = verbose
        self.DSR_limit = DSR_limit

        self.actual_state, self.visible_state = self.load_data(self.episode_length, self.lag, self.random_ts)
        self.state_models = []

        self.asset_models = []

        self.state = self.reset()


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _step(self, action, horizion=4):
        actual_state = self.actual_state.iloc[self.steps, 1:]

        past_DSR_used = self.info[-max(1, len(self.info)-horizion):, 2].sum()
        DSR_available = DSR_MAX - past_DSR_used
        DSR_used = min(action, DSR_capacity_available)

        price = 40
        reward = price * DSR_capacity_used

        self.info.append([action,
                          DSR_available,
                          DSR_used])

        self.steps += int(1)
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True

        next_state = self.visible_state.iloc[self.steps, 1:] # visible state
        self.state = next_state

        self.action_space = self.create_action_space()

        return next_state, reward, self.done, self.info

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Non-Open AI methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _load_data(self, episode_length, lag, random_ts):

        return visible_state, actual_state

    def _create_action_space(self):

        return action_space
