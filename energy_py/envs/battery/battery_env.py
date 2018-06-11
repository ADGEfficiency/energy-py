import logging
from random import random
import numpy as np

from energy_py.envs import BaseEnv
from energy_py.scripts.spaces import ContinuousSpace, GlobalSpace

logger = logging.getLogger(__name__)


class Battery(BaseEnv):
    """
    An environment that simulates storage of electricity in a battery.
    Agent chooses to either charge or discharge.

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

        #  technical energy inputs
        #  initial charge is set during reset()
        self.power_rating = float(power_rating)  # MW
        self.capacity = float(capacity)  # MWh
        self.round_trip_eff = float(round_trip_eff)  # %
        self.initial_charge = initial_charge

        #  initializing the BaseEnv class
        super().__init__(**kwargs)

        """
        SETTING THE ACTION SPACE

        the action space has a single dimension, ranging from max charge
        to max discharge

        i.e. for a 2 MW battery
        -2 <-> 2
        """
        self.action_space = GlobalSpace([ContinuousSpace(-self.power_rating, 
                                                         self.power_rating)])

        """
        SETTING THE OBSERVATION SPACE
        """
        #  append on any additional variables we want our agent to see
        spaces = [ContinuousSpace(0, self.capacity)]
        space_labels = ['C_charge_level_[MWh]']

        #  create a energy_py GlobalSpace object for the observation space
        #  the Space and labels for the observation loaded from
        #  csv are automatically added on in make_observation_space
        self.observation_space = self.make_observation_space(spaces,
                                                             space_labels)

        #  set the initial observation by resetting the environment
        self.observation = self.reset()

    def __repr__(self):
        return '<energy_py BATTERY environment - {} MW {} MWh>'.format(
            self.power_rating, self.capacity)

    def _reset(self):
        """
        Resets the environment

        returns
            observation (np.array) the initial observation
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

        self.state = self.get_state(steps=self.steps)

        #  charge is passed as a list because 0 will evaluate to falsey
        self.observation = self.get_observation(steps=self.steps,
                                                append=[self.charge])

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
        #  pulling out the state infomation
        elect_price_index = self.state_info.index('C_electricity_price_[$/MWh]')
        electricity_price = self.state[0][elect_price_index]

        old_charge = self.charge

        #  our action is sent to the environment as (1, num_actions)
        assert action.shape == (1, 1)

        #  we pull out the action to make the code below cleaner
        action = action[0][0]

        #  checking the action is valid
        assert action >= self.action_space.low
        assert action <= self.action_space.high

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

        #  set a tolerance for the energy balances
        tolerance = 1e-10
        #  energy balance
        assert (self.charge) - (old_charge + net_stored) < tolerance
        #  check that our net_rate and net_stored are consistent
        assert (net_rate) - (12 * net_stored) < tolerance

        #  now we can calculate the reward
        #  the reward is simply the cost to charge
        #  or the benefit from discharging
        #  note that we use the gross rate, this is the effect on the site
        #  import/export
        reward = -(gross_rate / 12) * electricity_price

        logger.debug('step is {:.3f}'.format(self.steps))
        logger.debug('action was {}'.format(action))
        logger.debug('old charge was {:.3f} MWh'.format(old_charge))
        logger.debug('new charge is {:.3f} MWh'.format(self.charge))
        logger.debug('gross rate is {:.3f} MW'.format(gross_rate))
        logger.debug('losses were {:.3f} MWh'.format(losses))
        logger.debug('net rate is {:.3f} MW'.format(net_rate))
        logger.debug('reward is {:.3f} $/5min'.format(reward))

        self.steps += 1
        next_state = self.get_state(self.steps)
        next_observation = self.get_observation(self.steps,
                                                append=[float(self.charge)])

        #  check to see if episode is done
        #  -1 in here because of the zero index
        if self.steps == (self.episode_length-1):
            self.done = True

        #  saving info
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

        #  moving to next time step
        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info


if __name__ == '__main__':
    import energy_py
    batt = energy_py.make_env('Battery')

    a = batt.action_space.sample()

    o, r, d, i = batt.step(a)
