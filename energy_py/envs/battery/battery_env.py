import logging
from random import random

from energy_py.envs import BaseEnv
from energy_py.scripts.spaces import ContinuousSpace, GlobalSpace

logger = logging.getLogger(__name__)


class Battery(BaseEnv):
    """
    An environment that simulates storage of electricity in a battery.
    Agent chooses to either charge or discharge.

    args
        data_path (str) location of state.csv, observation.csv
        episode_length (int)
        episode_start (int) integer index of episode start
        episode_random (bool) whether to randomize the episode start position
        power_rating (float) maximum rate of battery charge or discharge [MW]
        capacity (float) amount of electricity that can be stored [MWh]
        round_trip_eff (float) round trip efficiency of storage [%]
        initial_charge (float or str) inital charge as pct of capacity [%]
                               also possible to pass 'random'

    """
    def __init__(self,
                 data_path,
                 episode_length=48,
                 episode_start=0,
                 episode_random=False,
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

        super().__init__(data_path,
                         episode_length,
                         episode_start,
                         episode_random,
                         **kwargs)
        """
        SETTING THE ACTION SPACE

        two actions
         1 -  how much to charge [MWh]
         2 -  how much to discharge [MWh]

        use two actions to keep the action space positive
        is useful for policy gradient where we take log(probability of action)
        """
        self.action_space = GlobalSpace([ContinuousSpace(0, self.power_rating),
                                         ContinuousSpace(0, self.power_rating)])

        """
        SETTING THE OBSERVATION SPACE

        the observation space is set in the parent class TimeSeriesEnv
        we also append on an additional observation of the battery charge
        """
        #  make a list of the observation spaces
        observation_space, self.observation_ts, self.state_ts = self.get_state_obs()

        #  append on any additional variables we want our agent to see
        observation_space.append(ContinuousSpace(0, self.capacity))
        self.observation_info.append('C_charge_level_[MWh]')

        #  create a energy_py GlobalSpace object for the observation space
        self.observation_space = GlobalSpace(observation_space)

        #  set the initial observation by resetting the environment
        self.observation = self.reset()

    def __repr__(self):
        repr = '<energy_py BATTERY environment - {} MW {} MWh>'
        return repr.format(self.power_rating, self.capacity)

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

        initial_charge = float(self.capacity * initial_charge)  # MWh

        #  reseting the step counter, state, observation & done status
        self.steps = 0
        self.done = False

        self.state = self.get_state(steps=self.steps,
                                    append=initial_charge)

        self.observation = self.get_observation(steps=self.steps,
                                                append=initial_charge)

        #  pull the charge out of the state variable to check it
        initial_charge = self.state[0][-1]
        assert initial_charge <= self.capacity
        assert initial_charge >= 0

        logger.info('resetting environment')
        logger.debug('initial state is {}'.format(self.state))
        logger.debug('initial obs is {}'.format(self.observation))
        logger.debug('initial charge is {}'.format(initial_charge))

        return self.observation

    def _step(self, action):
        """
        One step through the environment.

        Battery is charged or discharged according to
        the action.

        args
            action (np.array) shape=(1, 2)
                          [0][0] = charging
                          [0][1] = discharging
        returns
            observation (np.array) shape=(1, len(self.observation_space)
            reward (float)
            done (boolean)
            info (dictionary)
        """
        #  pulling out the state infomation
        elect_price_index = self.state_info.index('C_electricity_price_[$/MWh]')
        electricity_price = self.state[0][elect_price_index]
        old_charge = self.state[0][-1]

        #  our action is sent to the environment as (1, num_actions)
        assert action.shape == (1, 2)

        #  we pull out the action here to make the code below cleaner
        action = action[0]

        #  checking the actions are valid
        for i, act in enumerate(action):
            assert act >= self.action_space.spaces[i].low
            assert act <= self.action_space.spaces[i].high

        #  calculate the net effect of the two actions
        #  also convert from MW to MWh/5 min by /12
        net_charge = (action[0] - action[1]) / 12

        #  we first check to make sure this charge is within our capacity limits
        unbounded_new_charge = old_charge + net_charge
        bounded_new_charge = max(min(unbounded_new_charge, self.capacity), 0)

        #  we can now calculate the gross rate of charge or discharge
        gross_rate = (bounded_new_charge - old_charge) * 12

        #  now we account for losses / the round trip efficiency
        if gross_rate > 0:
            #  we lose electricity when we charge
            losses = gross_rate * (1 - self.round_trip_eff) / 12
        else:
            #  we don't lose anything when we discharge
            losses = 0

        #  we can now calculate the new charge of the battery after losses
        new_charge = old_charge + gross_rate / 12 - losses
        #  this allows us to calculate how much electricity we actually store
        net_stored = new_charge - old_charge
        #  and to calculate our actual rate of charge or discharge
        net_rate = net_stored * 12

        #  set a tolerance for the energy balances
        tolerance = 1e-10
        #  energy balance
        assert (new_charge) - (old_charge + net_stored) < tolerance
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
        logger.debug('new charge is {:.3f} MWh'.format(new_charge))
        logger.debug('gross rate is {:.3f} MW'.format(gross_rate))
        logger.debug('losses were {:.3f} MWh'.format(losses))
        logger.debug('net rate is {:.3f} MW'.format(net_rate))
        logger.debug('reward is {:.3f} $/5min'.format(reward))

        self.steps += 1
        next_state = self.get_state(self.steps, append=float(new_charge))
        next_observation = self.get_observation(self.steps, append=float(new_charge))

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
                                     new_charge=new_charge,
                                     old_charge=old_charge,
                                     net_stored=net_stored)

        #  moving to next time step
        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info
