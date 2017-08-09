import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_py.envs.env_core import Base_Env, Continuous_Space


class Battery_Env(Base_Env):
    """
    An environment that simulates storage of electricity in a battery.
    Agent chooses to either charge or discharge.

    Args:
        lag                     (int)   : lag between observation & state
        episode_length          (int)   : length of the episdode

        power_rating            (float) : rate of battery to charge or discharge [MWe]
        capacity                (float) : amount of electricity that can be stored [MWh]
        round_trip_eff          (float) : round trip efficiency of storage
                                          how much of the stored electricity we
                                          can later extract
        initial_charge          (float) : inital amount of electricity stored [MWh]
        verbose                 (int)   : controls env print statements
    """
    def __init__(self, lag,
                       episode_length,
                       power_rating,
                       capacity,
                       round_trip_eff = 0.8,
                       initial_charge = 0,
                       verbose = 0):

        #  calling init method of the parent Base_Env class
        super().__init__()

        #  inputs relevant to the RL learning problem
        self.lag            = lag
        self.episode_length = episode_length

        #  technical energy inputs
        self.power_rating   = power_rating
        self.capacity       = capacity
        self.round_trip_eff = round_trip_eff
        self.initial_charge = initial_charge
        self.verbose        = verbose

        #  resetting the environment
        self.observation    = self.reset()

    def get_tests(self):
        return None

    def get_state(self, steps, charge):
        """
        Helper function to create the state numpy array

        Args:
            steps  (int)   : the relevant step for the desired state
            charge (float) : the charge to be appended to the state array
        """
        state_ts = np.array(self.state_ts.iloc[steps, :])
        state = np.append(state_ts, charge)
        return state

    def get_observation(self, steps, charge):
        """
        Helper function to create the state numpy array

        Args:
            steps (int) : the relevant step for the desired observation
        """
        observation_ts = np.array(self.observation_ts.iloc[steps, :])
        observation = np.append(observation_ts, charge)
        return observation

    def _reset(self):
        """
        Resets the environment.
        """
        #  we define our action space
        #  single action - how much to charge or discharge [MWh]

        self.action_space = [Continuous_Space(low  = -self.power_rating,
                                              high = self.power_rating,
                                              step = 1)]

        #  loading the state time series data
        csv_path = os.path.join(os.path.dirname(__file__), 'state.csv')
        self.observation_ts, self.state_ts = self.load_state(csv_path,
                                                             self.lag)

        #  defining the observation spaces from the state csv
        #  these are defined from the loaded csvs
        self.observation_space = [Continuous_Space(col.min(), col.max(), 1)
                                  for name, col in self.observation_ts.iteritems()]

        #  we also append on an additional observation of the battery charge
        self.observation_space.append(Continuous_Space(0, self.capacity, 1))

        #  setting the reward range
        self.reward_range = (-np.inf, np.inf)

        #  reseting the step counter, state, observation & done status
        self.steps = 0
        self.state = self.get_state(steps=self.steps, charge=self.initial_charge)
        self.observation = self.get_observation(steps=self.steps, charge=self.initial_charge)
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
        Args:
            action (np.array)         :
            where - action[0] (float) : rate to charge/discharge this time step
        """

        #  check that the action is valid
        for i, act in enumerate(action):
            assert self.action_space[i].contains(act), "%r (%s) invalid" % (action, type(action))

        #  pulling out the state infomation
        electricity_price = self.state[0]
        electricity_demand = self.state[1]
        old_charge = self.state[2]

        #  taking the action
        #  note we / 12 to convert from MW to MWh/5 min
        action = action[0]
        net_charge = action / 12
        unbounded_new_charge = old_charge + net_charge

        #  we first check to make sure this charge is within our capacity limits
        bounded_new_charge = max(min(unbounded_new_charge, self.capacity), 0)

        #  now we check to see this new charge is within our power rating
        #  note the * 12 is to convert from MWh/5min to MW
        #  here I am assuming that the power_rating is independent of charging/discharging
        unbounded_rate = (bounded_new_charge - old_charge) * 12
        rate = max(min(unbounded_rate, self.power_rating), -self.power_rating)

        #  finally we account for round trip efficiency
        losses = 0

        if rate > 0:
            losses = rate * (1 - self.round_trip_eff) / 12

        new_charge = old_charge + rate / 12 - losses
        net_stored = new_charge - old_charge
        rate = net_stored * 12

        assert new_charge == old_charge + net_stored
        assert net_stored * 12 == rate

        assert rate <= self.power_rating
        assert rate >= -self.power_rating

        assert new_charge <= self.capacity
        assert new_charge >= 0

        #  calculate the business as usual cost
        #  BAU depends on
        #  - site demand
        #  - electricity price
        BAU_cost = (electricity_demand / 12) * electricity_price

        #  now we can calculate the reward
        #  reward depends on both
        #  - how much electricity the site is demanding
        #  - what our battery is doing
        #  - electricity price
        adjusted_demand = electricity_demand + rate / 12
        RL_cost = (adjusted_demand / 12) * electricity_price
        reward = - RL_cost

        #  getting the next state & next observation
        next_state = self.get_state(self.steps + 1, charge=new_charge)
        next_observation = self.get_observation(self.steps + 1, charge=new_charge)

        #  saving info
        self.info = self.update_info(steps                   = self.steps,
                                     state                   = self.state,
                                     observation             = self.observation,
                                     action                  = action,
                                     reward                  = reward,
                                     next_state              = next_state,
                                     next_observation        = next_observation,
                                     BAU_cost                = BAU_cost,
                                     RL_cost                 = RL_cost,
                                     electricity_price       = electricity_price,
                                     electricity_demand = electricity_demand,
                                     rate                    = rate,
                                     new_charge = new_charge)

        if self.verbose > 0:
            print('step is {}'.format(self.steps))
            print('action was {}'.format(action))
            print('old charge was {}'.format(old_charge))
            print('new charge is {}'.format(new_charge))
            print('rate is {}'.format(rate))
            print('losses were {}'.format(losses))

        #  check to see if episode is done
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True
        else:
        #  moving onto next step
            self.steps += int(1)
            self.state = next_state
            self.observation = next_observation

        return self.observation, reward, self.done, self.info

    def update_info(self, steps,
                          state,
                          observation,
                          action,
                          reward,
                          next_state,
                          next_observation,

                          BAU_cost,
                          RL_cost,

                          electricity_price,
                          electricity_demand,
                          rate,
                          new_charge):
        """
        helper function to updates the self.info dictionary
        """
        self.info['steps'].append(steps)
        self.info['state'].append(state)
        self.info['observation'].append(observation)
        self.info['action'].append(action)
        self.info['reward'].append(reward)
        self.info['next_state'].append(next_state)
        self.info['next_observation'].append(next_observation)

        self.info['BAU_cost'].append(BAU_cost)
        self.info['RL_cost'].append(RL_cost)

        self.info['electricity_price'].append(electricity_price)
        self.info['electricity_demand'].append(electricity_demand)
        self.info['rate'].append(rate)
        self.info['new_charge'].append(new_charge)

        return self.info

    def output_info(self):
        """
        extracts info and turns into dataframes & graphs
        """
        def time_series_fig(df, cols, xlabel, ylabel):
            """
            makes a time series figure from a dataframe and specified columns
            """
            #  make the figure & axes objects
            fig, ax = plt.subplots(1, 1, figsize = (20, 20))
            for col in cols:
                df.loc[:, col].plot(kind='line', ax=ax, label=col)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            return fig

        RL_cost = sum(self.info['RL_cost'])
        BAU_cost = sum(self.info['BAU_cost'])
        print('RL cost was {}'.format(RL_cost))
        print('BAU cost was {}'.format(BAU_cost))
        print('Savings were {}'.format(BAU_cost-RL_cost))

        self.outputs['dataframe'] = pd.DataFrame.from_dict(self.info)
        self.outputs['dataframe'].index = self.state_ts.index[:len(self.outputs['dataframe'])]
        self.outputs['dataframe'].to_csv('output_df.csv')

        self.outputs['technical_fig'] = time_series_fig(df=self.outputs['dataframe'],
                                                          cols=['rate',
                                                                'new_charge',
                                                                'action'],
                                                          ylabel='Electricity [MW or MWh]',
                                                          xlabel='Time')

        self.outputs['cost_fig'] = time_series_fig(df=self.outputs['dataframe'],
                                                          cols=['BAU_cost',
                                                                'RL_cost',
                                                                'electricity_price'],
                                                          ylabel='Cost to deliver electricity [$/hh]',
                                                          xlabel='Time')
        return self.outputs
