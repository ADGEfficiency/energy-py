""" v3 of a price responsive flexible electricity asset

Add units onto all of the info dict lists

TODO
- test suite, both unit tests of the actions and a test_expt

test_no_op

"""

from collections import deque
import logging

import numpy as np

from energy_py.envs import BaseEnv
from energy_py.common import ContinuousSpace, DiscreteSpace, GlobalSpace


logger = logging.getLogger(__name__)


class Flex(BaseEnv):

    def __init__(self,
                 capacity=2.0,      # MWh
                 release_time=4, # num 5 mins
                 **kwargs):

        self.capacity = float(capacity)
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

        #  let our agent see the stored energy 
        self.observation_space.extend(
            ContinuousSpace(0, self.capacity), 'C_charge_level [MWh]'
        )

    def __repr__(self):
        return '<energy_py flex environment>'

    def _reset(self):
        """
        Resets the environment

        returns
           observation (np.array) the initial observation
        """
        self.steps = 0
        self._charge = 0

        #  storage history holds MWh per 5 mins
        self.storage_history = deque(maxlen=self.release_time)
        [self.storage_history.append(0) for _ in range(self.release_time)]

        self.state = self.state_space(self.steps)

        self.observation = self.observation_space(
            self.steps, np.array(self.charge)
        )

        return self.observation

    @property
    def charge(self):
        return self._charge

    @charge.getter
    def charge(self):
        return sum(self.storage_history)

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
        action = action[0][0]
        site_demand = self.get_state_variable('C_demand [MW]') / 12

        #  no-op
        if action == 0:
            stored = 0

            discharged = self.storage_history.pop()
            self.storage_history.appendleft(stored)

        #  cooling setpoint increased
        if action == 1:
            discharged = self.storage_history.pop()

            spare_capacity = self.capacity - self.charge
            stored = np.min([spare_capacity, site_demand])
            self.storage_history.appendleft(stored)

        #  cooling setpoint decreased
        elif action == 2:
            stored = 0

            discharged = sum(self.storage_history)
            [self.storage_history.appendleft(0)
             for _ in range(self.storage_history.maxlen)]
            assert self.charge == 0

        net = stored - discharged
        site_elecricity_consumption = site_demand - net

        electricity_price = self.get_state_variable('C_electricity_price [$/MWh]')
        reward = - (site_elecricity_consumption / 12) * electricity_price

        next_state = self.state_space(self.steps + 1)

        next_observation = self.observation_space(
            self.steps + 1, np.array(self.charge)
        )

        self.steps += 1

        done = False
        if self.steps == (self.state_space.episode.shape[0] - 1):
            done = True

        #  +1 for increased setpoint, zero for no op, -1 for decreased setpoint
        if action == 2:
            setpoint = -1
        else:
            setpoint = action

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
            'charge': self.charge,
            'site_demand': site_demand,
            'stored': stored,
            'discharged': discharged,
            'site_electricity_consumption': site_elecricity_consumption,
            'setpoint': setpoint,
            'demand_delta': -net
                }

        self.info = self.update_info(**info)
        [logger.debug('{} {}'.format(k, v)) for k, v in info.items()]

        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, done, self.info
