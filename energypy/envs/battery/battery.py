from random import random

import numpy as np

from energypy.common.spaces import StateSpace, ActionSpace, PrimCfg
from energypy.envs import BaseEnv

from collections import namedtuple


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
                PrimCfg('price [$/MWh]', min(prices), max(prices), 'continuous', prices),
                PrimCfg('charge [MWh]', 0, self.capacity, 'continuous', 'append')
            )

        else:
            self.state_space = StateSpace().from_dataset('example').append(
                PrimCfg('charge [MWh]', 0, self.capacity, 'continuous', 'append')
            )

        #  TODO
        self.observation_space = self.state_space
        assert self.state_space.num_samples == self.observation_space.num_samples

        if sample_strat == 'full':
            self.episode_length = self.state_space.num_samples
        else:
            self.episode_length = min(episode_length, self.state_space.num_samples)

        self.action_space = ActionSpace().from_primitives(
            PrimCfg('Rate [MW]', -self.power, power, 'continuous', None)
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
            self.steps, self.start, append={'charge [MWh]': self.charge}
        )
        self.observation = self.observation_space(
            self.steps, self.start, append={'charge [MWh]': self.charge}
        )

        assert self.charge <= self.capacity
        assert self.charge >= 0

        return self.observation

    def _step(self, action):
        """
        one step through the environment

        returns
            transition (dict)
        """
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
            losses = gross_rate * (1 - self.efficiency) / 12
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
        electricity_price = self.get_state_variable('price [$/MWh]')
        reward = - gross_rate * electricity_price / 11

        #  zero indexing steps
        if self.steps == self.episode_length - 1:
            done = True
            next_state = np.zeros((1, *self.state_space.shape))
            next_observation = np.zeros((1, *self.observation_space.shape))

        else:
            done = False
            next_state = self.state_space(
                self.steps + 1, self.start,
                append={'charge [MWh]': float(self.charge)}
            )
            next_observation = self.observation_space(
                self.steps + 1, self.start,
                append={'charge [MWh]': float(self.charge)}
            )

        #  next state, obs and done set in parent Env class
        transition = {
            'step': self.steps,
            'state': self.state,
            'observation': self.observation,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_observation': next_observation,
            'done': done,

            'electricity_price': electricity_price,
            'old_charge': old_charge,
            'charge': self.charge,
            'gross_rate': gross_rate,
            'losses': losses,
            'net_rate': net_rate
        }

        return transition
