""" This naive agent takes actions used predefined rules.

A naive agent is useful as a baseline for comparing with reinforcement
learning agents.

As the rules are predefined each agent is specific to an environment.
"""
import logging
import numpy as np

from energy_py.agents import BaseAgent
logger = logging.getLogger(__name__)


import pdb

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




class ClassifierAgent(BaseAgent):
    """
    Flexes based on a prediction from a classifier.
    """

    def __init__(self, env, discount, **kwargs):

        super().__init__(env, discount, memory_length=10)

    def find_index(self, h, c):
        """
        Helper function to find the index of a predicition in the observation
        numpy array

        args
            h (int) horizion
            c (str) class or bin
        """
        string = 'D_h_{}_Predicted_Price_Bin_{}'.format(h, c)

        return self.observation_info.index(string)

    def _act(self, observation):
        """

        notes
            perhaps the agent should also look at the minute
            i.e. not taking an action after 5 min into the hh
        """
        #  default action is to do nothing
        action = 0

        #  first a set of strageties to do the flex DOWN then UP cycle

        #  if current HH is VERY HIGH
        cond1 = observation[0, self.find_index(0, 'Very High')] == 1
        #  and we also need the forecast after to not be very high or high
        cond3 = observation[0, self.find_index(6, 'Very High')] == 0

        if cond1 and cond3:
            logger.info('taking an DOWN - UP action')
            logger.info('very high cond')
            action = 1

        #  if current HH is HIGH and next is less than HIGH
        cond2 = observation[0, self.find_index(0, 'High')] == 1
        cond4 = observation[0, self.find_index(6, 'High')] == 0

        # if cond2 and cond3 and cond4:
        #     logger.info('taking an DOWN - UP action')
        #     logger.info('high cond')
        #     action = 1

        # minute = observation[0, self.observation_info.index('C_minute')]
        # if (minute == 0 or minute == 30) and action != 0:
        #     logger.info('timing ok')
        #     action = action
        # else:
        #     logger.info('too late')
        #     action = 0

        return np.array(action).reshape(1, self.action_space.shape[0])

    def _learn(self, **kwargs):
        pass
