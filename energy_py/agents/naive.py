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
        idx = self.env.observation_info.index(
            'C_cumulative_mean_dispatch_[$/MWh]'
        )
        cumulative_dispatch = obs[0][idx]

        if cumulative_dispatch > self.trigger:
            action = self.action_space.high

        else:
            action = self.action_space.low

        return np.array(action).reshape(1, self.action_space.shape[0])


class TimeFlex(BaseAgent):
    """
    Flexes based on time of day

    args
        hours (list) hours to flex in

    kwargs passed into BaseAgent
        env (energy_py environment)
    """
    def __init__(self, hours, **kwargs):
        super().__init__(**kwargs)
        assert repr(self.env) == '<energy_py flex-v0 environment>'

        #  can be used for a two block period
        #  hours is input in the form
        #  [start, end, start, end]

        #  if block catches the case when we use config files, and hours
        #  iteratble ends up being a string like '5','9','15','19'
        if isinstance(hours, str):
            hours = hours.split(',')
            hours = [int(h) for h in hours]

        assert len(hours) == 4

        self.hours = np.concatenate([
            np.arange(hours[0], hours[1]),
            np.arange(hours[2], hours[3]),
        ])

        logging.info('time flex hours are {}'.format(hours))

        self.hour_index = self.env.observation_info.index('C_hour')

    def _act(self, observation):
        hour = observation[0][self.hour_index]

        if hour in self.hours:
            #  1 because we want up then down
            action = np.array(1)
        else:
            action = np.array(0)

        logging.debug('hour {} action {}'.format(hour, action))

        return np.array(action).reshape(1, self.action_space.shape[0])


class AutoFlex(BaseAgent):
    """
    Flexes based on the price predictions for the current and next hh

    kwargs passed into BaseAgent
        env (energy_py environment)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert repr(self.env) == '<energy_py flex-v0 environment>'

        self.minute_index = self.env.observation_info.index('C_minute')

        self.current_fc_index = self.env.observation_info.index(
            'C_forecast_electricity_price_current_hh_[$/MWh]')

        self.next_fc_index = self.env.observation_info.index(
            'C_forecast_electricity_price_next_hh_[$/MWh]')

    def _act(self, observation):
        minute = observation[0][self.minute_index]
        current_price = observation[0][self.current_fc_index]
        next_price = observation[0][self.next_fc_index]

        action = 0
        if minute == 0 or minute == 30:
            price_delta = current_price - next_price
            if price_delta > 5:
                #  1 becuase we wnat down then up
                action = 1

        logging.debug('minute {} current_p {} next_p {} action {}'.format(
            minute, current_price, next_price, action)
                      )

        return np.array(action).reshape(1, self.action_space.shape[0])




















class RandomAgent(BaseAgent):
    """
    An agent that always randomly samples the action space.

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
