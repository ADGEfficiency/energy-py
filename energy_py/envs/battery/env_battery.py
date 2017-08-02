import collections

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
    """
    def __init__(self, lag,
                       episode_length,
                       power_rating,
                       capacity,
                       round_trip_eff = 0.8,
                       initial_charge = 0):

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
        observation = np.array(self.observation_ts.iloc[steps, :])
        return observation

    def _reset(self):
        """
        Resets the environment.
        """
        #  we define our action space
        #  single action - how much to charge or discharge [MWh]

        self.action_space = [Continuous_Space(low  = -self.capacity, 
                                              high = self.capacity,
                                              step = 1)]

        #  loading the state time series data
        self.observation_ts, self.state_ts = self.load_state('state.csv',
                                                             self.lag,
                                                             self.episode_length)

        #  defining the observation spaces from the state csv
        #  these are defined from the loaded csvs
        self.observation_space = [Continuous_Space(col.min(), col.max())
                                  for name, col in self.observation_ts.iteritems()]

        #  we also append on an additional observation of the battery charge
        self.observation_space.append(Continuous_Space(0, self.capacity))

        #  reseting the step counter, state, observation & done status
        self.steps = 0
        self.state = self.get_state(steps=self.steps, charge=self.initial_charge)
        self.observation = self.get_observation(steps=self.steps, charge=self.initial_charge)
        self.done  = False

        initial_charge = self.state[2]
        assert initial_charge <= self.capacity
        assert initial_charge >= 0

        #  resetting the info & outputs dictionaries
        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        return self.observation

    def _step(self, action):
        """
        Args:
            action [net_charge] : action to perform this time step
        """
        print('step is {}'.format(self.steps))

        #  check that the action is valid
        for i, act in enumerate(action):
            assert self.action_space[i].contains(act), "%r (%s) invalid" % (action, type(action))

        #  pulling out the state infomation
        electricity_price = self.state[0]
        site_electricity_demand = self.state[1]
        old_charge = self.state[2]

        #  taking the actions
        net_charge = action 
        unbounded_new_charge = old_charge + net_charge

        #  we first check to make sure this charge is within our capacity limits
        bounded_new_charge = max(min(unbounded_new_charge, self.capacity), 0)

        #  now we check to see this new charge is within our power rating
        #  note the * 2 is to convert from MWh/hh to MW
        #  here I am assuming that the power_rating is independent of charging/discharging
        unbounded_rate = (bounded_new_charge - old_charge) * 2
        rate = max(min(unbounded_rate, self.power_rating), -self.power_rating)

        # / 2 is to convert from MW to MWh/hh
        new_charge = old_charge + rate / 2
        net_stored = new_charge - old_charge

        assert rate <= self.power_rating
        assert rate >= -self.power_rating

        assert new_charge <= self.capacity
        assert new_charge >= 0

        assert net_stored * 2 == rate

        #  finally we account for the round trip efficiency
        #  accounted for by electricity being lost as soon as it's stored
        losses = 0
        if new_charge > old_charge:
            losses = net_stored * self.round_trip_eff

        new_charge = new_charge - losses
        net_stored = new_charge - old_charge
        assert new_charge == old_charge + net_stored + losses

        #  calculate the business as usual cost
        #  BAU depends on
        #  - site demand
        #  - electricity price
        BAU_cost = (site_electricity_demand / 2) * electricity_price

        #  now we can calculate the reward
        #  reward depends on both
        #  - how much electricity the site is demanding
        #  - what our battery is doing
        #  - electricity price
        adjusted_demand = site_electricity_demand - rate / 2
        RL_cost = (adjusted_demand / 2) * electricity_price
        reward = - RL_cost

        #  getting the next state & next observation
        next_state = self.get_state(self.steps + 1, charge=new_charge)
        next_observation = self.get_observation(self.steps + 1, charge=new_charge)

        #  saving info
        self.info = self.update_info(steps            = self.steps,
                                     state            = self.state,
                                     observation      = self.observation,
                                     action           = action,
                                     reward           = reward,
                                     next_state       = next_state,
                                     next_observation = next_observation,
                                     BAU_cost         = BAU_cost,
                                     RL_cost          = RL_cost)

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
                          RL_cost):
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
        self.outputs['dataframe'].index = self.state_ts.index
        self.outputs['dataframe'].to_csv('output_df.csv')
        #
        # self.outputs['cooling_demand_fig'] = time_series_fig(df=self.outputs['dataframe'],
        #                                                   cols=['cooling_demand',
        #                                                         'adjusted_demand'],
        #                                                   ylabel='Cooling Demand [MW]',
        #                                                   xlabel='Time')

        self.outputs['cost_fig'] = time_series_fig(df=self.outputs['dataframe'],
                                                          cols=['BAU_cost',
                                                                'RL_cost',
                                                                'electricity_price'],
                                                          ylabel='Cost to deliver electricity [$/hh]',
                                                          xlabel='Time')
        return self.outputs


if __name__ == '__main__':
    env = Battery_Env(lag=0,
                      episode_length=48,
                      power_rating=2,
                      capacity=10)


    for i in range(env.episode_length):
        action = [1, 0]
        if i > 30:
            action=[0,1]
        env.step(action)

    outputs = env.output_info()
    f1 = outputs['cooling_demand_fig']
    f1.show()

    f2 = outputs['cost_fig']
    f2.show()
