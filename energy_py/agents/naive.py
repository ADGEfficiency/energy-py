""" This naive agent takes actions used predefined rules.

A naive agent is useful as a baseline for comparing with reinforcement
learning agents.

As the rules are predefined each agent is specific to an environment.
"""
import logging

import numpy as np

from energy_py.agents.agent import BaseAgent


logger = logging.getLogger(__name__)


class NaiveBatteryAgent(BaseAgent):
    """
    Charges at max/min based on the hour of the day.
    """

    def __init__(self, **kwargs):
        """
        args
            env (object)
        """
        #  calling init method of the parent Base_Agent class
        super().__init__(**kwargs)

        #  find the integer index of the hour in the observation
        self.hour_index = self.observation_info.index('D_hour')

    def _act(self, observation):
        """

        """
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
    def __init__(self, trigger=200, **kwargs):
        super().__init__(**kwargs)

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

    def __init__(self, hours, run_weekend=False, **kwargs):
        """
        args
            env (object)
            hours (list) hours to flex in
            run_weekend (bool)
        """
        #  calling init method of the parent Base_Agent class
        super().__init__(**kwargs)

        #  can be used for a two block period
        #  hours is input in the form
        #  [start, end, start, end]
        assert len(hours) == 4

        self.hours = np.concatenate([
            np.arange(hours[0], hours[1]),
            np.arange(hours[2], hours[3]),
        ])

        #  find the integer index of the hour in the observation
        self.hour_index = self.env.observation_info.index('C_hour')

    def _act(self, observation):
        """

        """
        #  index the observation at 0 because observation is
        #  shape=(num_samples, observation_length)
        hour = observation[0][self.hour_index]

        if hour in self.hours:
            action = self.action_space.high

        else:
            action = self.action_space.low

        return np.array(action).reshape(1, self.action_space.shape[0])


class RandomAgent(BaseAgent):
    """
    An agent that samples the action space.

    args
        env (object) energy_py environment
    """
    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

    def _act(self, observation):
        """
        Agent selects action randomly

        returns
             action (np.array)
        """
        return self.action_space.sample()


if __name__ == '__main__':
    import energy_py

    env = energy_py.make_env(
        'Flex-v1',
        flex_size=1,
        max_flex_time=4,
        relax_time=0,
        dataset='tempus')

    a = energy_py.make_agent('naive_flex', env=env, hours=(6, 10, 15, 19))

    o = env.reset()
    done = False
    while not done:
        action = a.act(o)

        o, r, done, i = env.step(action)

        print(action)
