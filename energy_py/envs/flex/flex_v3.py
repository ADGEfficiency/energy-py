""" v3 of a price responsive flexible electricity asset 

TODO
- test suite, both unit tests of the actions and a test_expt 

"""

from collections import deque
import logging

import numpy as np

from energy_py.envs import BaseEnv
from energy_py.common import ContinuousSpace, DiscreteSpace, GlobalSpace


logger = logging.getLogger(__name__)

class FlexV3(BaseEnv):

    def __init__(self,
                 capacity=2,   # MWh
                 release_time=24, # num 5 minute periods
                 **kwargs):

        self.capacity = capacity
        self.release_time = release_time

        super().__init__(**kwargs)

        """
        SETTING THE ACTION SPACE

        Single action - whether to start the flex asset or not
            0 = do nothing
            1 = store energy
            2 = release energy
        """
        self.action_space = GlobalSpace([DiscreteSpace(3)])

        """
        SETTING THE OBSERVATION SPACE

        Append the current amount of stored energy

        """
        self.observation_space = self.make_observation_space(
            [ContinuousSpace(0, self.capacity)],
            ['stored_energy']
        )

        self.observation = self.reset()

    def __repr__(self):
        return '<energy_py flex-v3 environment>'

    def _reset(self):
        """
        Resets the environment

        returns
           observation (np.array) the initial observation
        """
        self.steps = 0

        #  storage history holds MWhs
        self.storage_history = deque(maxlen=self.release_time)

        self.state = self.get_state(self.steps)
        self.observation = self.get_observation(
            self.steps,
            append=[self.stored_energy]
        )

        return self.observation

    @property
    def stored_energy(self):
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
        assert action.shape == (1, 1)
        action = action[0][0]

        assert action >= self.action_space.low
        assert action <= self.action_space.high

        site_demand = self.get_state_variable('C_demand [MW]') / 12
        electricity_price = self.get_state_variable('C_electricity_price [$/MWh]')

        if action == 1:
            #  cooling setpoint is raised
             stored = np.clip(old_charge + site_demand, 0, self.capacity)
             discharged = self.storage_history.pop(stored)
             net = stored - discharged

        elif action == 2:
            #  cooling setpoint is reduced
            stored=0
            discharged = sum(
                [self.storage_history.pop(0)
                 for _ in range(self.release_time)]
            )

            assert sum(self.storage_history) == 0

        site_elecricity_consumption = site_demand - stored + released

        reward = -(site_elecricity_consumption / 12) * electricity_price

        logger.debug('step {:.2f}'.format(self.steps))
        logger.debug('action {}'.format(action))
        logger.debug('old charge was {:.3f} MWh'.format(old_charge))
        logger.debug('new charge is {:.3f} MWh'.format(self.charge))
        logger.debug('gross rate is {:.3f} MW'.format(gross_rate))
        logger.debug('losses were {:.3f} MWh'.format(losses))
        logger.debug('net rate is {:.3f} MW'.format(net_rate))
        logger.debug('reward is {:.3f} $/5min'.format(reward))


if __name__ == '__main__':
    env = FlexV3()


