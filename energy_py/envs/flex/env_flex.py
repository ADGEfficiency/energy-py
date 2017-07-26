import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_py.envs.env_core import Base_Env, Discrete_Space, Continuous_Space

class Flexibility_Env(Base_Env):
    """
    An environment that simulates a flexibile electricity system

    The environment simulates a pre-cooling type demand flexibility action

    Pre-cooling occurs for a given amount of time (memory)
    After pre-cooling is finished post-cooling occurs
    Finally there is a user defined amount of relaxation time between the
    end of the post-cooling and the start of the next pre-cooling

    Args:
        lag             (int) : lag between observation & state
        episode_length  (int) : length of the episdode
        memory          (int) : number of time steps pre-cooling & post-cooling events last for
        relaxation_time (int) : time between end of post-cool & next pre-cool
        COP             (int) : a coefficient of performance for the chiller
    """

    def __init__(self, lag,
                       episode_length,
                       cooling_adjustment_time,
                       relaxation_time,
                       COP):

        #  calling init method parent class
        super().__init__()

        #  inputs relevant to the RL learning problem
        self.lag = lag
        self.episode_length = episode_length
        #  technical energy inputs
        self.cooling_adjustment_time = cooling_adjustment_time
        self.relaxation_time = relaxation_time
        self.COP = COP

        #  resetting the environment
        self.state = self.reset()

    def get_tests(self):
        return None

    def _reset(self):
        """
        resets the environment
        """
        #  we define our action space
        #  it's a single action - a binary start pre-cooling now or not
        self.action_space = [Discrete_Space(low  = 0,
                                            high = 1,
                                            step = 1)]

        #  loading the state time series data
        self.observation_ts, self.state_ts = self.load_state('state.csv',
                                                             self.lag,
                                                             self.episode_length)

        #  defining the observation spaces
        self.observation_space = [Continuous_Space(col.min(), col.max())
                                  for name, col in self.observation_ts.iteritems()]

        self.test_state_actions = self.get_tests()

        #  setting to the initial state
        self.state = self.state_ts.iloc[0, :]
        #  reseting the step counter
        self.steps = 0
        self.done = False
        #  resetting the deques used to track history
        self.precool_hist = collections.deque([], maxlen=self.cooling_adjustment_time)
        self.postcool_hist = collections.deque([], maxlen=self.cooling_adjustment_time)
        self.relaxation_hist = collections.deque([], maxlen=self.relaxation_time)
        #  resetting the info dictionary
        self.info['cooling_demand'] = []
        self.info['electricity_price'] = []
        self.info['action'] = []
        self.info['action'] = []
        self.info['demand_adjustment_hist'] = []
        self.info['precool_hist'] = []
        self.info['postcool_hist'] = []
        self.info['relaxation_hist'] = []
        self.info['steps'] = []
        self.info['adjusted_demand'] = []
        #  resetting the outputs dictionary
        self.outputs = {}
        print('environment reset')
        return self.state

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

        cooling_demand = self.state.loc['cooling_demand']
        electricity_price = self.state.loc['electricity_price']

        #  summing the history of the pre-cooling, post-cooling & relaxation modes
        precool_sum = sum(self.precool_hist)
        postcool_sum = sum(self.postcool_hist)
        relaxation_sum = sum(self.relaxation_hist)

        precool_count = sum([1 for x in self.precool_hist if x != 0])
        postcool_count = sum([1 for x in self.postcool_hist if x != 0])
        relaxation_count = sum([1 for x in self.relaxation_hist if x != 0])

        print(precool_sum)
        print(postcool_sum)
        print(relaxation_sum)
        print('precool count {}'.format(precool_count))
        print('postcool count {}'.format(postcool_count))
        print('relax count {}'.format(relaxation_count))

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
        #  -
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
        reward = adjusted_demand * electricity_price / self.COP

        #  saving info
        self.info = self.update_info(cooling_demand, electricity_price, action, demand_adjustment, adjusted_demand)

        #  check to see if episode is done
        if self.steps == (self.episode_length - abs(self.lag) - 1):
            self.done = True
        else:
        #  moving onto next step
            self.steps += int(1)
            self.state = self.state_ts.iloc[self.steps, :]

        return self.state, reward, self.done, self.info

    def update_hists(self, precool, postcool, relaxation):
        """
        helper function to update the deques
        """
        self.precool_hist.append(precool)
        self.postcool_hist.append(postcool)
        self.relaxation_hist.append(relaxation)
        return None

    def update_info(self, cooling_demand, electricity_price, action, demand_adjustment, adjusted_demand):
        """
        helper function to updates the self.info dictionary
        """
        self.info['steps'].append(self.steps)
        self.info['cooling_demand'].append(cooling_demand)
        self.info['electricity_price'].append(electricity_price)
        self.info['action'].append(action)
        self.info['demand_adjustment_hist'].append(demand_adjustment)
        self.info['adjusted_demand'].append(adjusted_demand)
        self.info['precool_hist'].append(self.precool_hist[-1])
        self.info['postcool_hist'].append(self.postcool_hist[-1])
        self.info['relaxation_hist'].append(self.relaxation_hist[-1])
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

        cooling_demand = sum(self.info['cooling_demand'])
        adjusted_demand = sum(self.info['adjusted_demand'])
        print('total cooling demand was {}'.format(cooling_demand))
        print('total adjusted demand was {}'.format(adjusted_demand))

        self.outputs['dataframe'] = pd.DataFrame.from_dict(self.info)
        self.outputs['dataframe'].index = self.state_ts.index
        self.outputs['dataframe'].to_csv('output_df.csv')

        self.outputs['time_series_fig'] = time_series_fig(df=self.outputs['dataframe'],
                                                          cols=['cooling_demand',
                                                                'adjusted_demand'],
                                                          ylabel='Cooling Demand [MW]',
                                                          xlabel='Time')
        return self.outputs


if __name__ == '__main__':
    env = Flexibility_Env(lag=0,
                          episode_length=48,
                          cooling_adjustment_time=4,
                          relaxation_time=48,
                          COP=1)

    for i in range(env.episode_length):
        action = 0
        if i > 24:
            action=1
        env.step(action)

    outputs = env.output_info()
    fig = outputs['time_series_fig']
    fig.show()
    df = outputs['dataframe']
    print(df.describe())
