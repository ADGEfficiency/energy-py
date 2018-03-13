import random

import gym
import numpy as np

from energy_py.envs.flex.env_flex import Flex
from energy_py.envs.battery.battery_env import Battery


class EnvWrapper(object):

    def __init__(self, env):
        self.env = env
        self.observation_info = None

    def __repr__(self):
        return repr(self.env)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def discretize(self, num_discrete):
        self.actions = list(self.action_space.discretize(num_discrete))
        return self.actions

    def sample_discrete(self):
        return self.env.action_space.sample_discrete()


class FlexEnv(EnvWrapper):

    def __init__(self, **kwargs):
        env = Flex(**kwargs)
        super(FlexEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape
        self.observation_info = self.env.observation_info
        self.action_space = self.env.action_space
        self.action_space_shape = self.action_space.shape

        self.observation_info = env.observation_info


class BatteryEnv(EnvWrapper):

    def __init__(self, **kwargs):
        env = Battery(**kwargs)
        super(BatteryEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.observation_info = self.env.observation_info
        self.action_space = self.env.action_space
        self.action_space_shape = self.action_space.shape

        self.observation_info = env.observation_info


class CartPoleEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('CartPole-v0')
        super(CartPoleEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space_shape = (1,)

    def step(self, action):
        #  cartpole doesn't accept an array!
        return self.env.step(action[0][0])

    def discretize(self, num_discrete):
        self.actions = [np.array(act) for act in range(self.action_space.n)]
        return self.actions

    def sample_discrete(self):
        return random.choice(self.actions)


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
        self.actions = np.linspace(self.action_space.low,
                                   self.action_space.high,
                                   num=num_discrete,
                                   endpoint=True).tolist()
        return self.actions


class MountainCarEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('Pendulum-V0')
        super(MountainCarEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space_shape = (1,)

    def discretize(self, num_discrete):
        self.actions = [act for act in range(self.action_space.n)]
        return self.actions
