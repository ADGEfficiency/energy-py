import pandas as pd

from energy_py.envs.env_core import Base_Env, Discrete_Space, Continuous_Space

class Flexibility_Env(Base_Env):
    """
    An environment that simulates a flexibile electricity system

    The environment simulates a pre-cooling type demand flexibility action

    Args:
        capacity (float) : amount of electricity shifted during a pre-cooling event
        memory   (int)   : number of time steps pre-cooling & postcooling event lasts for
        lag      (int)   : lag between observation & state
    """

    def __init__(self, capacity,
                       memory,
                       lag):

        self.capacity = capacity
        self.memory = memory
        self.lag = lag

        #  we define our action space
        #  it's a single action - a binary start pre-cooling now or not
        self.action_space = Discrete_Space(low  = 0,
                                           high = 1,
                                           step = 1)

        #  we define our observation space
        #  our observation space is two dimensions
        #   1 = price [$/MWh]
        #   2 = customer demand [MW]
        self.observation_ts, self.state_ts, self.observation_space = self.load_state()
        self.test_state_actions = self.get_tests()

        #  resetting the environment
        self.state = self.reset()

    def load_state(self):
        #  our state infomation is a time series
        #  loading time series data
        ts = pd.read_csv('state.csv',
                         index_col=0,
                         parse_dates=True)

        #  now dealing with the lag
        #  if no lag then state = observation
        if self.lag == 0:
            observation_ts = ts.iloc[:, :]
            state_ts = ts.iloc[:, :]

        #  a negative lag means the agent can only see the past
        elif self.lag < 0:
            #  we shift & cut the observation
            observation_ts = ts.shift(self.lag).iloc[:-self.lag, :]
            #  we cut the state
            state_ts = ts.iloc[:-self.lag, :]

        #  a positive lag means the agent can see the future
        elif self.lag > 0:
            #  we shift & cut the observation
            observation_ts = ts.shift(self.lag).iloc[self.lag:, :]
            #  we cut the state
            state_ts = ts.iloc[self.lag:, :]

        #  checking our two ts are the same shape
        assert observation_ts.shape == state_ts.shape

        #  now we create our spaces objects
        observation_space = [Continuous_Space(col.min(), col.max())
                             for colname, col in observation_ts.iteritems()]

        return observation_ts, state_ts, observation_space

    def get_tests(self):
        return None

    def _reset(self):
        #  setting to the initial state
        self.state = self.state_ts.iloc[0, :]
        #  reseting the step counter
        self.steps = 0
        #  resetting the pre-cooling event history
        self.precool_history = collections.deque(maxlen=self.memory)

        return np.array(self.state)

    def _step(self, action):
        """
        Args:
            action (boolean) : whether or not to start a precooling event

        """
        #  check that the action is valid
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        site_demand = self.state_ts.loc['site_demand']
        price = self.state_ts.loc['electricity_price']


        #  check whether a post-cooling event is taking place



        self.steps += 1
        return np.array(self.state), reward, done, {}


if __name__ == '__main__':
    env = Flexibility_Env(capacity = 5,
                          memory   = 5,
                          lag      = 1)

    obs_space = env.observation_space
    action_space = env.action_space

    action = action_space.sample()
    print(action)
