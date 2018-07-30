""" v3 of a price responsive flexible electricity asset

TODO
Add units onto all of the info dict lists
"""

from collections import deque
import logging

import numpy as np

from energy_py.envs import BaseEnv
from energy_py.common import ContinuousSpace, DiscreteSpace, GlobalSpace


logger = logging.getLogger(__name__)


class Flex(BaseEnv):
    """
    ## demand side response model

    Asset can operate in four dimensions
    1. store demand (reducing site electricity consumption)
    2. release demand (increasing site electricity consumption)
    3. storing supply (increases site electricity consumption)
    4. releasing supply (decreases site electricity consumption)

    Storing and releasing demand is the classic use of demand side response
    where the control is based on reducing the asset electricity consumption

    Storing and releasing supply is the the inverse - controlling an asset by
    increasing electricity consumption - avoiding consuming electricity later

    ## structure of the agent

    Demand is stored and released from a deque.  A deque structure is used so that
    even if the agent keeps the setpoint raised, the demand will be released
    The time difference between store and release is the length of the deque

    Supply is stored using a float.  Because supply is a cost, the agent not using
    it by releasing is behaviour I want the agent to learn to avoid

    The structure of the agent is inspired by anecdotal observation of
    commerical chiller plants reacting to three different setpoints
    - increased (demand stored)
    - no_op
    - decreased (supply stored - i.e. precooling)

    """
    def __init__(
            self,
            capacity=4.0,         # MWh
            precool_capacity=0.5, # MWh
            release_time=12,      # num 5 mins
            precool_power=0.05,      # MW
            scale_reward=False,   #  bool - move to baseenv at somepoint TODO
            **kwargs
    ):

        self.capacity = float(capacity)
        self.supply_capacity = float(precool_capacity)

        self.release_time = int(release_time)
        self.precool_power = float(precool_power)

        self.scale_reward = scale_reward

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

        #  let our agent see the stored demand
        #  let our agent see the stored supply
        #  see precool power?

        #  obs space is created during env init
        self.observation_space.extend(
            [ContinuousSpace(0, self.capacity),
            ContinuousSpace(0, self.supply_capacity)],
            ['C_stored_demand [MWh]',
            'C_stored_supply[MWh]'],
        )

    def _reset(self):
        """
        Resets the environment

        returns
           observation (np.array) the initial observation
        """
        self.steps = 0

        self._charge = 0  #MWh

        #  use a deque for stored demand (all MWh per 5min)
        self.storage_history = deque(maxlen=self.release_time)
        [self.storage_history.appendleft(0) for _ in range(self.release_time)]

        #  float for stored supply
        self.stored_supply = 0  # MWh

        self.state = self.state_space(self.steps)

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
        """ args MW return MW """
        released = min(demand / 12, self.stored_supply)
        self.stored_supply -= released
        return released * 12

    def store_demand(self, demand):
        """ always store - check for the capacity done elsewhere """
        """ args MW return MW """
        self.storage_history.appendleft(demand / 12)

        logger.debug('storing {}'.format(self.storage_history))
        return demand

    def dump_demand(self):
        """ returns MW """
        dumped = sum(self.storage_history)

        [self.storage_history.appendleft(0)
         for _ in range(self.storage_history.maxlen)]

        assert self.stored_demand == 0

        return dumped * 12

    def store_supply(self, demand):
        """ args MW return MW """
        old_precool = self.stored_supply

        new_precool = np.clip(
            old_precool + (self.precool_power - demand) / 12,
            0,
            self.precool_capacity
        )

        precooling = new_precool - old_precool
        self.stored_supply += precooling

        return precooling * 12

    def _step(self, action):
        """
        One step through the environment

        args
            action (np.array) shape=(1, 1)

        returns
            observation (np.array) shape=(1, self.observation_space.shape*)
            reward (float)
            done (bool)
            info (dict)
        """
        action = action[0][0]  #  could do this in BaseAgent

        site_demand = self.get_state_variable('C_demand [MW]') / 12
        site_consumption = site_demand

        #  these can be simplified - unless info wanted for debug
        #  ie move from var = self.fun(). site_cons += var
        #  to site_cons += self.fun() etc

        released_demand = self.storage_history.pop()

        #  no-op
        if action == 0:
            released_supply = self.release_supply(site_consumption)
            self.storage_history.appendleft(0)
            # print('released supply {}'.format(released_supply))
            site_consumption -= released_supply

        #  raising setpoint (reducing demand)
        if action == 1:
            stored_demand = self.store_demand(site_consumption)
            site_consumption -= stored_demand
            # print('{} stored demand {} site_consumption'.format(
            #     stored_demand, site_consumption))

        site_consumption += released_demand * 12

        #  reducing setpoint (increasing demand)
        if action == 2:
            stored_demand_dump = self.dump_demand()
            site_consumption += stored_demand_dump

            precooling = self.store_supply(site_consumption)
            site_consumption += precooling

        #  dump out the entire stored demand if we reach capacity
        #  this is the chiller ramping up to full when return temp gets
        #  too high
        if self.stored_demand >= self.capacity:
            site_consumption += self.dump_demand()

        print('released demand {}'.format(released_demand))

        # if action == 1:
        #     print('test before save')
        #     print('{} stored demand {} site_consumption'.format(
        #         stored_demand, site_consumption))

        electricity_price = self.get_state_variable(
            'C_electricity_price [$/MWh]')
        baseline_cost = site_demand * electricity_price / 12
        optimized_cost = site_consumption * electricity_price / 12

        #  negative means we are increasing cost
        #  positive means we are reducing cost
        reward = baseline_cost - optimized_cost

        if self.scale_reward:
            scale_factor = self.state_space.data[:, 'C_electricity_price [$/MWh]'].max()

            scale_factor *= scale_factor * (self.precool_power + self.state_space.data[:, 'C_demand [MW]'].max())
            reward = reward / scale_factor

        next_state = self.state_space(self.steps + 1)

        next_observation = self.observation_space(
            self.steps + 1, np.array(
                [self.stored_demand, self.stored_supply]
            )
        )

        self.steps += 1

        done = False
        if self.steps == (self.state_space.episode.shape[0] - 1):
            done = True
            # TODO add in mechanism to dump out stored demand
            # at the end of the episode
            # not implementing now because I want to see if the agents learn
            # that they can store for free at the end (ie in the last few
            # steps of the episode, where steps_left < release_time)

        setpoint = 0
        if action == 1:
            setpoint = 1
        elif action == 2:
            setpoint = -1

        info = {
            'step': self.steps,
            'state': self.state,
            'observation': self.observation,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_observation': next_observation,
            'done': done,

            'electricity_price': electricity_price,
            'stored_demand': self.stored_demand,
            'stored_supply': self.stored_supply,
            'site_demand': site_demand,
            'site_consumption': site_consumption,
            'net_discharged': site_consumption - site_demand,
            'setpoint': setpoint,
                }

        self.info = self.update_info(**info)
        [logger.debug('{} {}'.format(k, v)) for k, v in info.items()]

        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, done, self.info
