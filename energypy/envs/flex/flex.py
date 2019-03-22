""" v3 of a price responsive flexible electricity asset

TODO
Add units onto all of the info dict lists
"""

from collections import deque
import logging

import numpy as np

from energypy.envs import BaseEnv
from energypy.common import ContinuousSpace, DiscreteSpace, GlobalSpace


logger = logging.getLogger(__name__)


class Flex(BaseEnv):
    """ price responsive flexible demand model """

    def __init__(
            self,
            capacity=4.0,         # MWh
            supply_capacity=0.5,  # MWh
            release_time=12,      # num 5 mins
            supply_power=0.05,    # MW
            **kwargs
    ):

        self.capacity = float(capacity)

        #  this should look at the max of the data - TODO
        self.supply_capacity = float(supply_capacity)

        self.release_time = int(release_time)

        super().__init__(**kwargs)

        """
        action space has a single discrete dimension
        0 = no op
        1 = increase setpoint
        2 = decrease setpoint
        """
        self.action_space = GlobalSpace('action').from_spaces(
            DiscreteSpace(3), 'setpoint'
        )

        self.action_space.no_op = np.array([0]).reshape(1, 1)

        self.state_space.extend(
            [ContinuousSpace(0, self.episode_length),
             ContinuousSpace(0, self.capacity),
             ContinuousSpace(0, self.supply_capacity)],
            ['Step', 'C_stored_demand [MWh]', 'C_stored_supply[MWh]'],
        )

        #  let our agent see the stored demand
        #  let our agent see the stored supply
        #  see precool power?

        #  obs space is created during env init
        self.observation_space.extend(
            [ContinuousSpace(0, self.capacity),
             ContinuousSpace(0, self.supply_capacity)],
            ['C_stored_demand [MWh]', 'C_stored_supply[MWh]'],
        )

        #  supply = precooling
        #  i.e how much power we consume during precooling
        self.supply_power = float(max(
            float(supply_power),
            self.state_space.data.loc[:, 'C_demand [MW]'].max()
        ))

    def __repr__(self):
        return '<energypy flex environment>'

    def _reset(self):
        """
        Resets the environment

        returns
           observation (np.array) the initial observation
        """
        self.steps = 0
        self._charge = 0  # MWh

        """
        Demand is stored and released from a deque.  A deque is used so that
        even if the agent keeps the setpoint raised, the demand will be released

        The time difference between store and release is the length of the deque

        The deque stores MWh/5 min
        """
        self.storage_history = deque(maxlen=self.release_time)
        [self.storage_history.appendleft(0) for _ in range(self.release_time)]

        #  use a float for the inverse of stored demand
        self.stored_supply = 0  # MWh

        self.state = self.state_space(
            self.steps, np.array(
                [self.steps, self.stored_demand, self.stored_supply]
            )
        )

        self.observation = self.observation_space(
            self.steps, np.array(
                [self.stored_demand, self.stored_supply]
            )
        )

        return self.observation

    @property
    def stored_demand(self):
        return self._stored_demand

    @stored_demand.getter
    def stored_demand(self):
        return sum(self.storage_history)

    def release_supply(self, demand):
        """ net off our demand with some stored supply """
        """ args MWh return MWh """
        released_supply = min(demand, self.stored_supply)
        self.stored_supply -= released_supply
        # self.storage_history.appendleft(0)  ?????
        demand -= released_supply
        return demand, released_supply

    def store_demand(self, stored_demand):
        """ always store - check for the capacity done elsewhere """
        """ args MWh return MWh """
        self.storage_history.appendleft(stored_demand)
        return 0, stored_demand

    def release_demand(self, demand):
        """ args MWh returns MWh """
        dumped = sum(self.storage_history)

        [self.storage_history.appendleft(0)
         for _ in range(self.storage_history.maxlen)]
        assert self.stored_demand == 0

        return demand - dumped, dumped

    def store_supply(self, demand):
        """ args MWh return MWh """
        old_stored_supply = self.stored_supply  # MWh

        #  here we are assuming that supply power can always be
        #  met in excess of whatever demand there is

        stored_supply = np.min(
            [self.supply_capacity - old_stored_supply,
            (self.supply_power / 12) - demand]
        )

        self.stored_supply += stored_supply

        return demand + stored_supply, stored_supply

    def _step(self, action):
        """ one step through the environment """
        action = action[0][0]  #  could do this in BaseAgent

        #  do everything in the MWh / 5 min space
        site_demand = self.get_state_variable('C_demand [MW]') / 12
        flexed = site_demand

        #  no-op
        if action == 0:
            setpoint = 0
            flexed, released_supply = self.release_supply(flexed)
            flexed, dumped = self.release_demand(flexed)

        #  raising setpoint (reducing demand)
        if action == 1:
            setpoint = 1
            flexed, stored_demand = self.store_demand(flexed)

        #  reducing setpoint (increasing demand)
        if action == 2:
            setpoint = -1
            flexed, stored_demand_dump = self.release_demand(flexed)
            flexed, stored_supply = self.store_supply(flexed)

        #  dump out the entire stored demand if we reach capacity
        #  this is the chiller ramping up to full when return temp gets
        #  too high
        if self.stored_demand >= self.capacity:
            flexed, dumped = self.release_demand(flexed)

        #  do the same if the episode is over - dump everything out
        if self.steps == self.state_space.episode.shape[0] - 1:
            flexed, dumped = self.release_demand(flexed)

        electricity_price = self.get_state_variable(
            'C_electricity_price [$/MWh]')
        baseline_cost = site_demand * electricity_price * 12
        flexed_cost = flexed * electricity_price * 12

        #  negative means we are increasing cost
        #  positive means we are reducing cost
        self.reward = baseline_cost - flexed_cost

        if self.steps == self.state_space.episode.shape[0] - 1:
            self.done = True
            next_state = np.zeros((1, *self.state_space.shape))
            self.next_observation = np.zeros((1, *self.observation_space.shape))

        else:
            next_state = self.state_space(
                self.steps + 1,
                np.array([self.steps + 1, self.stored_demand,
                          self.stored_supply])
            )

            self.next_observation = self.observation_space(
                self.steps + 1,
                np.array([self.stored_demand, self.stored_supply])
            )

        transition = {
            'step': self.steps,
            'state': self.state,
            'observation': self.observation,
            'action': action,
            'reward': self.reward,
            'next_state': next_state,
            'next_observation': self.next_observation,
            'done': self.done,

            'electricity_price': electricity_price,
            'stored_demand': self.stored_demand,
            'stored_supply': self.stored_supply,
            'site_demand': site_demand,
            'flexed': flexed,
            'net_discharged': flexed - site_demand,
            'setpoint': setpoint,
                }

        return transition
