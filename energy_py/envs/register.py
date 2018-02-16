import gym

from energy_py.envs.flex.env_flex import Flex
from energy_py.envs.battery.battery_env import Battery


class EnvWrapper(object):

    def __init__(self, env):
        self.env = env

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


class FlexEnv(EnvWrapper):

    def __init__(self, **kwargs):
        env = Flex(**kwargs)
        super(FlexEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape
        self.observation_info = self.env.observation_info
        self.action_space = self.env.action_space
        self.action_space_shape = self.action_space.shape

    def discretize(self, num_discrete):
        """
        Not every agent will need to do this
        """
        return list(self.action_space.discretize(num_discrete))

class BatteryEnv(EnvWrapper):

    def __init__(self, **kwargs):
        env = Battery(**kwargs)
        super(BatteryEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.observation_info = self.env.observation_info
        self.action_space = self.env.action_space
        print(self.observation_info)
        self.action_space_shape = self.action_space.shape

    def discretize(self, num_discrete):
        """
        Not every agent will need to do this
        """
        return list(self.action_space.discretize(num_discrete))

class CartPoleEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('CartPole-v0')
        super(CartPoleEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space_shape = (1,)

    def discretize(self, num_discrete):
        """
        Not every agent will need to do this
        """
        return [act for act in range(self.action_space.n)]


class PendulumEnv(EnvWrapper):

    def __init__(self, num_discrete):
        env = gym.make('Pendulum-V0')
        super(PendulumEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space_shape = self.action_space.shape

    def discretize(self, num_discrete):
        """
        Not every agent will need to do this
        """
        return np.linspace(self.action_space.low,
                                   self.action_space.high,
                                   num=num_discrete,
                                   endpoint=True).tolist()


class MountainCarEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('Pendulum-V0')
        super(MountainCarEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space_shape = (1,)

    def discretize(self, num_discrete):
        """
        Not every agent will need to do this
        """
        return [act for act in range(self.action_space.n)]


