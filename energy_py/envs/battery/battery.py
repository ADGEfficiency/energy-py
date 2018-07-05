import logging
from random import random

import numpy as np

from energy_py.common import ContinuousSpace, GlobalSpace
from energy_py.envs import BaseEnv

logger = logging.getLogger(__name__)


class Battery(BaseEnv):
    """
    Electric battery storage - rewarded by price arbitrage

    optionally passed into BaseEnv via kwargs
        dataset (str) located in energy_py/experiments/datasets
        episode_length (int)
        episode_start (int) integer index of episode start
        episode_random (bool) whether to randomize the episode start position

    args
        power_rating (float) maximum rate of battery charge or discharge [MW]
        capacity (float) amount of electricity that can be stored [MWh]
        round_trip_eff (float) round trip efficiency of storage [%]
        initial_charge (float or str) inital charge as pct of capacity [%]
                               also possible to pass 'random'
    """
    def __init__(self,
                 power_rating=2,
                 capacity=4,
                 round_trip_eff=0.9,
                 initial_charge=0.00,
                 **kwargs):

        self.power_rating = float(power_rating)  # MW
        self.capacity = float(capacity)  # MWh
        self.round_trip_eff = float(round_trip_eff)  # %
        self.initial_charge = initial_charge

        super().__init__(**kwargs)

        """
        action space has a single dimension, ranging from max charge
        to max discharge

        for a 2 MW battery, a range of -2 to 2 MW
        """
        self.action_space = GlobalSpace('action').from_spaces(
            [ContinuousSpace(-self.power_rating, self.power_rating)],
            'Rate [MW]'
        )

        self.observation_space.extend(ContinuousSpace(0, self.capacity),
                                      'C_charge_level [MWh]')

        self.observation = self.reset()

    def __repr__(self):
        return '<energy_py BATTERY environment - {} MW {} MWh>'.format(
            self.power_rating, self.capacity)

    def _reset(self):
        """
        Resets the environment

        returns
            observation (np.array) initial observation
        """
        #  setting the initial charge
        if self.initial_charge == 'random':
            initial_charge = random()  # %
        else:
            initial_charge = float(self.initial_charge)  # %

        self.charge = float(self.capacity * initial_charge)  # MWh

        #  reseting the step counter, state, observation & done status
        self.steps = 0
        self.done = False

        self.state = self.state_space(self.steps)
        self.observation = self.observation_space(
            self.steps,
            append=[self.charge]
        )

        #  pull the charge out of the state variable to check it
        assert self.charge <= self.capacity
        assert self.charge >= 0

        logger.debug('initial state is {}'.format(self.state))
        logger.debug('initial obs is {}'.format(self.observation))
        logger.debug('initial charge is {}'.format(self.charge))

        return self.observation

    def _step(self, action):
        """
        One step through the environment.

        Battery is charged or discharged according to
        the action.

        args
            action (np.array) shape=(1, 1)
                first dimension is the batch dimension - 1 for a single action
                second dimension is the charge 
                (-self.rating <-> self.power_rating)

        returns
            observation (np.array) shape=(1, len(self.observation_space)
            reward (float)
            done (boolean)
            info (dictionary)
        """
        assert action.shape == (1, 1)
        action = action[0][0]

        old_charge = self.charge

        #  convert from MW to MWh/5 min by /12
        net_charge = action / 12

        #  we first check to make sure this charge is within our capacity limit
        new_charge = np.clip(old_charge + net_charge, 0, self.capacity)

        #  we can now calculate the gross rate of charge or discharge
        gross_rate = (new_charge - old_charge) * 12

        #  now we account for losses / the round trip efficiency
        if gross_rate > 0:
            #  we lose electricity when we charge
            losses = gross_rate * (1 - self.round_trip_eff) / 12
        else:
            #  we don't lose anything when we discharge
            losses = 0

        #  we can now calculate the new charge of the battery after losses
        self.charge = old_charge + gross_rate / 12 - losses
        #  this allows us to calculate how much electricity we actually store
        net_stored = self.charge - old_charge
        #  and to calculate our actual rate of charge or discharge
        net_rate = net_stored * 12

        #  energy balances
        assert np.isclose(self.charge - old_charge - net_stored, 0)
        assert np.isclose(net_rate - net_stored * 12, 0)

        #  now we can calculate the reward
        #  the reward is simply the cost to charge
        #  or the benefit from discharging
        #  note that we use the gross rate, this is the effect on the site
        #  import/export
        electricity_price = self.get_state_variable(
            'C_electricity_price [$/MWh]')

        reward = - gross_rate * electricity_price / 12

        logger.debug('step is {:.3f}'.format(self.steps))
        logger.debug('action was {}'.format(action))
        logger.debug('old charge was {:.3f} MWh'.format(old_charge))
        logger.debug('new charge is {:.3f} MWh'.format(self.charge))
        logger.debug('gross rate is {:.3f} MW'.format(gross_rate))
        logger.debug('losses were {:.3f} MWh'.format(losses))
        logger.debug('net rate is {:.3f} MW'.format(net_rate))
        logger.debug('reward is {:.3f} $/5min'.format(reward))

        next_state = self.state_space(self.steps + 1)
        obs_append = self.make_observation_append_list()
        next_observation = self.observation_space(
            self.steps + 1, obs_append
        )

        self.steps += 1

        if self.steps == (self.state_space.episode.shape[0] - 1):
            self.done = True

        self.info = self.update_info(steps=self.steps,
                                     state=self.state,
                                     observation=self.observation,
                                     action=action,
                                     reward=reward,
                                     next_state=next_state,
                                     next_observation=next_observation,
                                     done=self.done,

                                     electricity_price=electricity_price,
                                     gross_rate=gross_rate,
                                     losses=losses,
                                     new_charge=self.charge,
                                     old_charge=old_charge,
                                     net_stored=net_stored)

        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info
