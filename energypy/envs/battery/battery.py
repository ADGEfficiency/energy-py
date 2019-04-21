from random import random

import numpy as np

from energypy.common.spaces import StateSpace, ActionSpace
from energypy.common.spaces import PrimitiveConfig as Prim
from energypy.envs import BaseEnv


class Battery(BaseEnv):
    """
    Electric battery operating in price arbitrage

    args
        power [MW]
        capacity [MWh]
        efficiency [%]
        initial_charge [% or 'random']
    """
    def __init__(
            self,
            power=2.0,
            capacity=4.0,
            efficiency=0.9,
            initial_charge=0.5,

            episode_length=2016,
            sample_strat='fixed',

            prices=None,
            dataset='example',
            **kwargs
    ):
        self.power = float(power)
        self.capacity = float(capacity)
        self.efficiency = float(efficiency)
        self.initial_charge = initial_charge
        self.sample_strat = sample_strat
        super().__init__(**kwargs)

        #  this is fucking messy
        if prices is not None:
            self.state_space = StateSpace().from_primitives(
                Prim('Price [$/MWh]', min(prices), max(prices), 'continuous', prices),
                Prim('Charge [MWh]', 0, self.capacity, 'continuous', 'append')
            )

        else:
            self.state_space = StateSpace().from_dataset(dataset).append(
                Prim('Charge [MWh]', 0, self.capacity, 'continuous', 'append')
            )

        #  TODO
        self.observation_space = self.state_space
        assert self.state_space.num_samples == self.observation_space.num_samples

        if sample_strat == 'full':
            self.episode_length = self.state_space.num_samples
        else:
            self.episode_length = min(episode_length, self.state_space.num_samples)

        self.action_space = ActionSpace().from_primitives(
            Prim('Power [MW]', -self.power, power, 'continuous', None)
        )

    def __repr__(self):
        return '<energypy BATTERY env - {:2.1f} MW {:2.1f} MWh>'.format(
            self.power, self.capacity)

    def _reset(self):
        """ samples new episode, returns initial observation """
        #  initial charge in percent
        if self.initial_charge == 'random':
            initial_charge = random()
        else:
            initial_charge = float(self.initial_charge)

        #  charge in MWh
        self.charge = float(self.capacity * initial_charge)

        #  new episode
        self.start, self.end = self.state_space.sample_episode(
            self.sample_strat, episode_length=self.episode_length
        )

        #  set initial state and observation
        self.state = self.state_space(
            self.steps, self.start, append={'Charge [MWh]': self.charge}
        )
        self.observation = self.observation_space(
            self.steps, self.start, append={'Charge [MWh]': self.charge}
        )

        assert self.charge <= self.capacity
        assert self.charge >= 0

        return self.observation

    def _step(self, action):
        """
        one step through the environment

        positive action = charging
        negative action = discharging

        returns
            transition (dict)
        """
        old_charge = self.charge

        #  convert from MW to MWh/5 min by /12
        net_charge = action / 12

        #  we first check to make sure this charge is within our capacity limit
        new_charge = np.clip(old_charge + net_charge, 0, self.capacity)

        #  we can now calculate the gross power of charge or discharge
        #  charging is positive
        gross_power = (new_charge - old_charge) * 12

        #  now we account for losses / the round trip efficiency
        if gross_power < 0:
            #  we lose electricity when we discharge
            losses = abs(gross_power * (1 - self.efficiency))

        else:
            #  we don't lose anything when we charge
            losses = 0

        net_power = gross_power + losses

        #  we can now calculate the new charge of the battery after losses
        self.charge = old_charge + (gross_power / 12)
        #  this allows us to calculate how much electricity we actually store
        net_stored = self.charge - old_charge

        #  energy balances
        assert np.isclose(gross_power - net_power, -losses)

        #  now we can calculate the reward
        #  the reward is simply the cost to charge
        #  or the benefit from discharging
        #  note that we use the gross power, this is the effect on the site
        #  import/export
        electricity_price = self.get_state_variable('Price [$/MWh]')
        reward = net_power * electricity_price / 12

        #  zero indexing steps
        if self.steps == self.episode_length - 1:
            done = True
            next_state = np.zeros((1, *self.state_space.shape))
            next_observation = np.zeros((1, *self.observation_space.shape))

        else:
            done = False
            next_state = self.state_space(
                self.steps + 1, self.start,
                append={'Charge [MWh]': float(self.charge)}
            )
            next_observation = self.observation_space(
                self.steps + 1, self.start,
                append={'Charge [MWh]': float(self.charge)}
            )

        #  next state, obs and done set in parent Env class
        transition = {
            'step': int(self.steps),
            'state': self.state,
            'observation': self.observation,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'next_observation': next_observation,
            'done': bool(done),

            'Price [$/MWh]': float(electricity_price),
            'Initial charge [MWh]': float(old_charge),
            'Final charge [MWh]': float(self.charge),
            'Gross [MW]': float(gross_power),
            'Net [MW]': float(net_power),
            'Loss [MW]': float(losses)
        }

        return transition
