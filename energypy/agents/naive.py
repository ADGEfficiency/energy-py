""" Naive agents used as baselines """
import logging

import numpy as np

from energy_py.agents.agent import BaseAgent


logger = logging.getLogger(__name__)


class NaiveBatteryAgent(BaseAgent):
    """ Charges at max/min based on the hour of the day """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hour_index = self.observation_space.info.index('D_hour')

    def _act(self, observation):
        hour = observation[0][self.hour_index]
        act_spaces = self.action_space.spaces

        #  discharge during morning peak
        if hour >= 7 and hour < 10:
            action = -self.env.power_rating

        #  discharge during evening peak
        elif hour >= 15 and hour < 21:
            action = -self.env.power_rating

        #  charge at max rate
        else:
            action = self.env.power_rating

        return np.array(action).reshape(1, self.action_space.shape[0])


class FeatureAgent(BaseAgent):
    """
    Takes an action based on the value of a feature

    args
        env (object) energy_py environment
        discount (float) discount rate
        trigget (float) triggers flex action based on the feature
    """
    def __init__(
            self,
            feature,
            feature_index,
            trigger,
            trigger_action,
            default_action,
            **kwargs
    ):
        self.feature = feature
        self.feature_index = feature_index

        self.trigger = float(trigger)
        self.trigger_action = trigger_action
        self.default_action = default_action

        super().__init__(**kwargs)

    def _act(self, observation):

        action = self.default_action
        if cumulative_dispatch > self.trigger:
            action = self.trigger_action

        return np.array(action).reshape(1, self.action_space.shape[0])


class TimeFlex(BaseAgent):
    """
    Flexes based on time of day

    args
        hours (list) hours to flex in
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
    """ Flexes based on the price predictions for the current and next hh """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert repr(self.env) == '<energy_py flex environment>'

        self.hh_tick_index = self.env.observation_space.info.index('D_hh_tick')

        self.current_fc_index = self.env.observation_space.info.index(
            'C_forecast_electricity_price_hh_0 [$/MWh]')

        self.next_fc_index = self.env.observation_space.info.index(
            'C_forecast_electricity_price_hh_1 [$/MWh]')

    def _act(self, observation, **kwargs):
        hh_tick = observation[0][self.hh_tick_index]
        current_price = observation[0][self.current_fc_index]
        next_price = observation[0][self.next_fc_index]

        action = 0
        delta = next_price - current_price

        #  if next price is lower, we want to not consume now (ie to store)
        if delta < -5 and hh_tick == 1:
            action = 1
            logger.debug('taking action - delta {}'.format(delta))

        else:
            logger.debug('no price delta {}'.format(delta))

        logger.debug('hh_tick {} current_p {} next_p {} delta {} action {}'.format(
            hh_tick, current_price, next_price, delta, action)
                      )

        return np.array(action).reshape(1, *self.action_space.shape)

    def _learn(self, *args, **kwargs):
        pass


class RandomAgent(BaseAgent):
    """ Randomly samples action space """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _act(self, observation, **kwargs):
        return self.action_space.sample()

    def _learn(self, *args, **kwargs):
        pass


class NoOp(BaseAgent):
    """ Does nothing each step """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _act(self, observation, **kwargs):
        return self.action_space.no_op

    def _learn(self, *args, **kwargs):
        pass
