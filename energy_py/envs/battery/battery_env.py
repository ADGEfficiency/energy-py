import collections
import os

import numpy as np

from energy_py.envs import Time_Series_Env
from energy_py.main.scripts.spaces import Continuous_Space, Discrete_Space


class Battery_Env(Time_Series_Env):
    """
    An environment that simulates storage of electricity in a battery.
    Agent chooses to either charge or discharge.

    args
        lag             :   int     : lag between observation & state
        episode_length  :   int     : length of the episode
                        :   string  : 'maximum' = run entire legnth
        episode_start   :   int     : the integer index to start the episode
        power_rating    :   float   : maximum rate of battery charge
                                      or discharge [MWe]
        capacity        :   float   : amount of electricity that can be stored
                                      [MWh]
        round_trip_eff  :   float   : round trip efficiency of storage [%]
        initial_charge  :   int     : inital amount of electricity stored
                                    : as a percent of capacity [%]

        verbose         :   int     : controls env print statements
    """
    def __init__(self, lag,
                       episode_length,
                       episode_start,

                       power_rating,
                       capacity,
                       round_trip_eff = 0.9,
                       initial_charge = 0,

                       verbose = False):

        path = os.path.dirname(os.path.abspath(__file__))
        state_path = os.path.join(path, 'state.csv')
        observation_path = os.path.join(path, 'observation.csv')

        print('state path is'.format(path))

        #  calling init method of the parent Time_Series_Env class
        super().__init__(lag,
                         episode_length,
                         episode_start,
                         state_path,
                         observation_path,
                         verbose)

        #  technical energy inputs
        self.power_rating   = float(power_rating)
        self.capacity       = float(capacity)
        self.round_trip_eff = float(round_trip_eff)
        self.initial_charge = float((self.capacity * initial_charge) / 100)

        #  resetting the environment
        self.observation    = self.reset()

    def _reset(self):
        """
        Resets the environment.
            
        returns
            observation : np.array : the initial observation
        """

        """
        SETTING THE ACTION SPACE

        two actions
         1 -  how much to charge [MWh]
         2 -  how much to discharge [MWh]

        use two actions to keep the action space positive
        is useful for policy gradient where we take log(probability of action)
        """
        self.action_space = [Continuous_Space(low  = 0,
                                              high = self.power_rating),
                             Continuous_Space(low  = 0,
                                              high = self.power_rating)]

        """
        SETTING THE OBSERVATION SPACE

        the observation space is set in the parent class Time_Series_Env
        we also append on an additional observation of the battery charge
        """
        self.observation_space, self.observation_ts, self.state_ts = self.get_state_obs()

        self.observation_space.append(Continuous_Space(0, self.capacity))

        """
        SETTING THE REWARD SPACE

        For the reward function
            reward = bau_cost - rl_cost

        maximum reward = full discharge at max electricity price
        minimum reward = full charge at max electricity price

        note that we take the maximum of the absolute electricity price
        """

        max_price = max(self.state_ts.loc[:, 'C_electricity_price_[$/MWh]'].abs())
        self.reward_space = Continuous_Space((-max_price * self.power_rating)/12,
                                             (max_price * self.power_rating)/12)

        #  reseting the step counter, state, observation & done status
        self.steps = 0
        self.state = self.get_state(steps=self.steps, append=self.initial_charge)
        self.observation = self.get_observation(steps=self.steps, append=self.initial_charge)
        self.done  = False

        initial_charge = self.state[-1]
        assert initial_charge <= self.capacity
        assert initial_charge >= 0

        #  resetting the info & outputs dictionaries
        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        return self.observation

    def _step(self, action):
        """
        One step through the environment.

        args
            action      : np.array (1, 2)
                          [0][0] = charging
                          [0][1] = discharging
        returns
            observation : np.array (1, len(self.observation_space)
            reward      : float
            done        : boolean
            info        : dictionary
        """

        #  pulling out the state infomation
        electricity_price = self.state[0]
        old_charge = self.state[-1]

        #  our action is sent to the environment as (1, num_actions)
        assert action.shape == (1, 2)

        #  we pull out the action here to make the code below cleaner
        action = action[0]

        #  checking the actions are valid
        for i, act in enumerate(action):
            assert act >= self.action_space[i].low
            assert act <= self.action_space[i].high

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

        self.verbose_print('step is {}'.format(self.steps),
                           'action was {}'.format(action),
                           'old charge was {} MWh'.format(old_charge),
                           'new charge is {} MWh'.format(new_charge),
                           'gross rate is {} MW'.format(gross_rate),
                           'losses were {} MWh'.format(losses),
                           'net rate is {} MW'.format(net_rate),
                           'reward is {} $/5min'.format(reward))

        #  check to see if episode is done
        #  -1 in here because of the zero index
        if self.steps == (self.episode_length-1):
            self.done = True
            next_state = 'terminal'
            next_observation = 'terminal'
            reward = 0

            total_ep_reward = sum(self.info['reward'])

            self.verbose_print('Episode {} done'.format(self.episode),
                               'total reward {}'.format(total_ep_reward))
        else:
        #  moving onto next step
            next_state = self.get_state(self.steps, append=float(new_charge))
            next_observation = self.get_observation(self.steps, append=float(new_charge))
            self.steps += 1

        #  saving info
        self.info = self.update_info(episode            = self.episode,
                                     steps              = self.steps,
                                     state              = self.state,
                                     observation        = self.observation,
                                     action             = action,
                                     reward             = reward,
                                     next_state         = next_state,
                                     next_observation   = next_observation,

                                     electricity_price  = electricity_price,
                                     gross_rate         = gross_rate,
                                     losses             = losses,
                                     new_charge         = new_charge,
                                     old_charge         = old_charge,
                                     net_stored         = net_stored)

        #  moving to next time step
        self.state = next_state
        self.observation = next_observation

        return self.observation, reward, self.done, self.info

    def update_info(self, episode,
                          steps,
                          state,
                          observation,
                          action,
                          reward,
                          next_state,
                          next_observation,

                          electricity_price,
                          gross_rate,
                          losses,
                          new_charge,
                          old_charge,
                          net_stored):
        """
        Helper function to update the self.info dictionary
        """
        self.info['episode'].append(episode)
        self.info['steps'].append(steps)
        self.info['state'].append(state)
        self.info['observation'].append(observation)
        self.info['action'].append(action)
        self.info['reward'].append(reward)
        self.info['next_state'].append(next_state)
        self.info['next_observation'].append(next_observation)

        self.info['electricity_price'].append(electricity_price)
        self.info['gross_rate'].append(gross_rate)
        self.info['losses'].append(losses)
        self.info['new_charge'].append(new_charge)
        self.info['old_charge'].append(old_charge)
        self.info['net_stored'].append(net_stored)

        return self.info

    def _output_results(self):
        """
        """
        self.outputs['dataframe'].loc[:,'rolling_rate']=self.outputs['dataframe'].loc[:,'gross_rate'].rolling(window=10,
                                                                                                         min_periods=1,
                                                                                                         center=False)
        return self.outputs
