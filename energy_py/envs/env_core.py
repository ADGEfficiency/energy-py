import collections
import logging
import os

import numpy as np
import pandas as pd


from energy_py import Utils

class BaseEnv(Utils):
    """
    the energy_py base environment class
    inspired by the gym.Env class

    The methods of this class are:
        step
        reset

    To implement an environment:
      - override the following methods in your child:
        _step()
        _reset()

      - set the following attributes
        action_space
        observation_space
        reward_range (defaults to -inf, +inf)

    args:
        verbose : boolean : controls printing
    """

    def __init__(self):
        super().__init__()
        self.observation = self.reset(episode='none')
        return None

    # Override in ALL subclasses
    def _step(self, action): raise NotImplementedError
    def _reset(self): raise NotImplementedError
    def _output_results(self): raise NotImplementedError

    #  Set these in ALL subclasses
    action_space = None       #  list of length num_actions
    observation_space = None  #  list of length obs_dim
    reward_space = None       #  single space object

    def reset(self, episode):
        """
        Resets the state of the environment and returns an initial observation.

        Returns: observation (np array): the initial observation
        """
        self.episode = episode
        logging.info('Reset environment')

        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list) 

        return self._reset()

    def step(self, action):
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

        args
            action  (object): an action provided by the environment
            episode (int): the current episode number

        returns:
            observation (np array): agent's observation of the current environment
            reward (np float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        return self._step(action)

    def output_results(self):
        self.outputs['dataframe'] = pd.DataFrame.from_dict(self.info)
        return self._output_results()


    def update_info(self, **kwargs):
        """
        Helper function to update the self.info dictionary.

        Use kwargs to give flexibility to the environment as to what to store.
        """
        for name, data in **kwargs.items():
            self.info[name].append(data)

        return self.info
