import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_py.envs.env_core import Base_Env, Discrete_Space, Continuous_Space

class Precool_Env(Base_Env):
    """
    An environment that simulates a pre-cooling type demand flexibility action.
    Agent chooses to start a pre-cooling action or not.

    Pre-cooling occurs for a given amount of time (cooling_adjustment_time).
    After pre-cooling is finished post-cooling occurs.

    Total cooling generated is the same for the RL and BAU case.

    Finally there is a user defined amount of relaxation time between the
    end of the post-cooling and the start of the next pre-cooling.

    Args:
        lag                     (int) : lag between observation & state
        episode_length          (int) : length of the episode

        cooling_adjustment_time (int) : number of time steps pre-cooling & post-cooling events last for
        relaxation_time         (int) : time between end of post-cool & next pre-cool
        COP                     (int) : a coefficient of performance for the chiller
    """

    def __init__(self, lag,
                       episode_length,
                       cooling_adjustment_time,
                       relaxation_time,
                       COP=3):

        #  calling init method of the parent Base_Env class
        super().__init__()

        #  inputs relevant to the RL learning problem
        self.lag = lag
        self.episode_length = episode_length

        #  technical energy inputs
        self.cooling_adjustment_time = cooling_adjustment_time
        self.relaxation_time = relaxation_time
        self.COP = COP

        #  resetting the environment
        self.observation = self.reset()

    def get_tests(self):
        return None

    def get_state(self, steps):
        """
        Helper function to create the state numpy array

        Args:
            steps (int) : the relevant step for the desired state
        """
        state = np.array(self.state_ts.iloc[steps, :])
        return state

    def get_observation(self, steps):
        """
        Helper function to create the state numpy array

        Args:
            steps (int) : the relevant step for the desired observation
        """
        observation = np.array(self.observation_ts.iloc[steps, :])
        return observation

    def _reset(self):
        """
        Resets the environment
        """
        #  we define our action space
        #  it's a single action - a binary start pre-cooling now or not
        self.action_space = [Discrete_Space(low  = 0,
                                            high = 1,
                                            step = 1)]

        #  loading the state time series data
        csv_path = os.path.join(os.path.dirname(__file__), 'state.csv')
        self.observation_ts, self.state_ts = self.load_state(csv_path,
                                                             self.lag)

        #  defining the observation spaces
        #  these are defined from the loaded csvs
        self.observation_space = [Continuous_Space(col.min(), col.max(), step=1)
                                  for name, col in self.observation_ts.iteritems()]

        #  reseting the step counter, state, observation & done status
        self.steps = 0
        self.state = self.get_state(self.steps)
        self.observation = self.get_observation(self.steps)
        self.done = False

        #  resetting the deques used to track history
        self.precool_hist = collections.deque([], maxlen=self.cooling_adjustment_time)
        self.postcool_hist = collections.deque([], maxlen=self.cooling_adjustment_time)
        self.relaxation_hist = collections.deque([], maxlen=self.relaxation_time)

        #  resetting the info & outputs dictionaries
        self.info = collections.defaultdict(list)
        self.outputs = collections.defaultdict(list)

        return self.observation

    def _step(self, action):
        """
        Args:
            action (boolean) : whether or not to start a precooling event
        """
        print('step is {}'.format(self.steps))
        print('pre-cooling history {}'.format(self.precool_hist))
        print('post-cooling history {}'.format(self.postcool_hist))
        print('relaxation history {}'.format(self.relaxation_hist))

        #  check that the action is valid
        assert self.action_space[0].contains(action), "%r (%s) invalid" % (action, type(action))

        #  pulling out the state infomation
        electricity_price = self.state[0]
        cooling_demand = self.state[1]

        #  summing the history of the pre-cooling, post-cooling & relaxation modes
        precool_sum = sum(self.precool_hist)
        postcool_sum = sum(self.postcool_hist)
        relaxation_sum = sum(self.relaxation_hist)

        precool_count = sum([1 for x in self.precool_hist if x != 0])
        postcool_count = sum([1 for x in self.postcool_hist if x != 0])
        relaxation_count = sum([1 for x in self.relaxation_hist if x != 0])

        #  should we start a precooling event
        #  - has an action started?
        #  - are we not already in a precool event
        #  - are we not in a relaxation period
        if action == 1 and precool_sum == 0 and relaxation_count == 0 and self.postcool_hist[-1] == 0:
            print('starting precooling')
            demand_adjustment = -cooling_demand
            self.update_hists(demand_adjustment, 0, 0)

        #  are we in a pre-cooling event
        #  - sum of pre-cool adjustments isn't zero
        #  - count of pre-cool adjustments is less than the cooling adjustment time
        #  - last entry of the post-cool isn't zero (ie a pre-cool hasn't just ended)
        elif precool_sum != 0 and precool_count < self.cooling_adjustment_time and self.precool_hist[-1] != 0:
            print('in pre-cooling')
            demand_adjustment = -cooling_demand
            self.update_hists(demand_adjustment, 0, 0)

        #  are we finishing a pre-cooling event / starting post-cooling
        #  - count of pre-cool is equal to the cooling adjustment time
        elif precool_count == self.cooling_adjustment_time:
            print('ending pre-cooling & starting post-cooling')
            demand_adjustment = -self.precool_hist[0]
            self.update_hists(0, demand_adjustment, 0)

        #  are we in a post-cooling event
        #  - sum of post-cool isn't zero
        #  - count of post-cool is less than the cooling adjustment time
        #  - last entry of the post-cool isn't zero (ie a post-cool hasn't just ended)
        elif postcool_sum != 0 and postcool_count < self.cooling_adjustment_time and self.postcool_hist[-1] != 0:
            print('in post-cooling')
            demand_adjustment = -self.precool_hist[0]
            self.update_hists(0, demand_adjustment, 0)

        #  are we ending a postcooling event
        elif postcool_count == self.cooling_adjustment_time and precool_sum == 0:
            print('ending post-cooling event')
            demand_adjustment = 0
            self.update_hists(0, 0, 1)

        else:
            print('nothing is happening')
            demand_adjustment = 0
            self.update_hists(0, 0, 0)

            if relaxation_count > 0:
                print('in relaxation time')

        print('demand adjustment is {}'.format(demand_adjustment))
        adjusted_demand = cooling_demand + demand_adjustment

        #  now we can calculate the reward
        #  the reward signal is the cost to generate cooling
        #  note we divide by two to get $/hh (state is on HH basis)
        RL_cost = (adjusted_demand / 2) * electricity_price / self.COP
        reward = - RL_cost

        #  we also calculate the business as usual cost
        BAU_cost = (cooling_demand / 2) * electricity_price / self.COP

        #  getting the next state & next observation
        next_state = self.get_state(self.steps + 1)
        next_observation = self.get_observation(self.steps + 1)

        #  saving info
        self.info = self.update_info(steps            = self.steps,
                                     state            = self.state,
                                     observation      = self.observation,
                                     action           = action,
                                     reward           = reward,
                                     next_state       = next_state,
                                     next_observation = next_observation,
                                     BAU_cost         = BAU_cost,
                                     RL_cost          = RL_cost,

                                     cooling_demand = cooling_demand,
                                     electricity_price = electricity_price,
                                     demand_adjustment = demand_adjustment,
                                     adjusted_demand = adjusted_demand)

        #  check to see if episode is done
        #  else move onto next step
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True
        else:
            self.steps += int(1)
            self.state = next_state
            self.observation = next_observation

        return self.observation, reward, self.done, self.info

    def update_hists(self, precool, postcool, relaxation):
        """
        Helper function to update the deques.
        """
        self.precool_hist.append(precool)
        self.postcool_hist.append(postcool)
        self.relaxation_hist.append(relaxation)
        return None

    def update_info(self, steps,
                          state,
                          observation,
                          action,
                          reward,
                          next_state,
                          next_observation,

                          BAU_cost,
                          RL_cost,

                          cooling_demand,
                          electricity_price,
                          demand_adjustment,
                          adjusted_demand):
        """
        Helper function to update self.info.
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

        self.info['cooling_demand'].append(cooling_demand)
        self.info['electricity_price'].append(electricity_price)
        self.info['demand_adjustment_hist'].append(demand_adjustment)
        self.info['adjusted_demand'].append(adjusted_demand)

        self.info['precool_hist'].append(self.precool_hist[-1])
        self.info['postcool_hist'].append(self.postcool_hist[-1])
        self.info['relaxation_hist'].append(self.relaxation_hist[-1])

        return self.info

    def output_info(self):
        """
        Extracts from self.info and turns into DataFrames & figures.
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

        cooling_demand = sum(self.info['cooling_demand'])
        adjusted_demand = sum(self.info['adjusted_demand'])
        print('total cooling demand was {}'.format(cooling_demand))
        print('total adjusted demand was {}'.format(adjusted_demand))

        RL_cost = sum(self.info['RL_cost'])
        BAU_cost = sum(self.info['BAU_cost'])
        print('RL cost was {}'.format(RL_cost))
        print('BAU cost was {}'.format(BAU_cost))
        print('Savings were {}'.format(BAU_cost-RL_cost))

        self.outputs['dataframe'] = pd.DataFrame.from_dict(self.info)
        self.outputs['dataframe'].index = self.state_ts.index[:len(self.outputs['dataframe'])]
        self.outputs['dataframe'].to_csv('output_df.csv')

        self.outputs['cooling_demand_fig'] = time_series_fig(df=self.outputs['dataframe'],
                                                          cols=['cooling_demand',
                                                                'adjusted_demand'],
                                                          ylabel='Cooling Demand [MW]',
                                                          xlabel='Time')

        self.outputs['cost_fig'] = time_series_fig(df=self.outputs['dataframe'],
                                                          cols=['BAU_cost',
                                                                'RL_cost',
                                                                'electricity_price'],
                                                          ylabel='Cost to deliver electricity [$/HH]',
                                                          xlabel='Time')
        return self.outputs


