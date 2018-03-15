""" This naive agent takes actions used predefined rules.

A naive agent is useful as a baseline for comparing with reinforcement
learning agents.

As the rules are predefined each agent is specific to an environment.
"""
import logging
import pdb

import numpy as np

from energy_py.agents import BaseAgent


logger = logging.getLogger(__name__)


class NaiveBatteryAgent(BaseAgent):
    """
    Charges at max/min based on the hour of the day.
    """

    def __init__(self, env, discount):
        """
        args
            env (object)
            discount (float)
        """
        #  find the integer index of the hour in the observation
        self.hour_index = self.observation_info.index('D_hour')

        #  calling init method of the parent Base_Agent class
        super().__init__(env, discount)

    def _act(self, **kwargs):
        """

        """
        observation = kwargs['observation']
        #  index the observation at 0 because observation is
        #  shape=(num_samples, observation_length)
        hour = observation[0][self.hour_index]

        #  grab the spaces list
        act_spaces = self.action_space.spaces

        if hour >= 7 and hour < 10:
            #  discharge during morning peak
            action = [act_spaces[0].low, act_spaces[1].high]

        elif hour >= 15 and hour < 21:
            #  discharge during evening peak
            action = [act_spaces[0].low, act_spaces[1].high]

        else:
            #  charge at max rate
            action = [act_spaces[0].high, act_spaces[1].low]

        return np.array(action).reshape(1, self.action_space.shape[0])


class DispatchAgent(BaseAgent):
    """
    Dispatch agent looks at the cumulative mean of the within half hour
    dispatch price.  Takes action if this cumulative average is greater
    than the trigger price

    args
        env (object) energy_py environment
        discount (float) discount rate
        trigget (float) triggers flex action based on cumulative mean price
    """
    def __init__(self, env, discount, trigger=200):
        #  calling init method of the parent Base_Agent class
        super().__init__(env, discount)
        self.trigger = float(trigger)

    def _act(self, **kwargs):
        """

        """
        obs = kwargs['observation']
        idx = self.env.observation_info.index('C_cumulative_mean_dispatch_[$/MWh]')
        cumulative_dispatch = obs[0][idx]

        if cumulative_dispatch > self.trigger:
            action = self.action_space.high

        else:
            action = self.action_space.low

        return np.array(action).reshape(1, self.action_space.shape[0])


class NaiveFlex(BaseAgent):
    """
    Flexes based on time of day
    """

    def __init__(self, env, discount, hours, run_weekend=False):
        """
        args
            env (object)
            discount (float)
            hours (list) hours to flex in
        """
        self.hours = hours

        #  calling init method of the parent Base_Agent class
        super().__init__(env, discount)

        #  find the integer index of the hour in the observation
        self.hour_index = self.env.observation_info.index('C_hour')

    def _act(self, **kwargs):
        """

        """
        observation = kwargs['observation']
        #  index the observation at 0 because observation is
        #  shape=(num_samples, observation_length)
        hour = observation[0][self.hour_index]

        if hour in self.hours:
            action = self.action_space.high 

        else:
            #  do nothing 
            action = self.action_space.low 

        return np.array(action).reshape(1, self.action_space.shape[0])
