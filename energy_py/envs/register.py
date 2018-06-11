"""
A registery for environments supported by energy_py

Combination of native energy_py environments and wrapped gym environments
"""

import logging
import random

import gym
import numpy as np

from energy_py.common import GlobalSpace

from energy_py.envs.flex.flex_v0 import FlexV0
from energy_py.envs.flex.flex_v1 import FlexV1

from energy_py.envs.battery.battery_env import Battery


logger = logging.getLogger(__name__)


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

    def discretize_action_space(self, num_discrete):
        self.actions = self.action_space.discretize(num_discrete)
        return self.actions

    def sample_discrete_action(self):
        return self.env.action_space.sample_discrete()


class FlexEnvV0(EnvWrapper):

    def __init__(self, **kwargs):
        env = FlexV0(**kwargs)
        super(FlexEnvV0, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape
        self.observation_info = self.env.observation_info
        self.action_space = self.env.action_space
        self.action_space_shape = self.action_space.shape

        self.observation_info = env.observation_info

class FlexEnvV1(EnvWrapper):

    def __init__(self, **kwargs):
        env = FlexV1(**kwargs)
        super(FlexEnvV1, self).__init__(env)

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

        self.action_space.discretize = self.discretize_action_space

    def step(self, action):
        #  cartpole doesn't accept an array!
        return self.env.step(action[0][0])

    def discretize_action_space(self, num_discrete):
        actions = [act for act in range(self.action_space.n)]

        #  cheat a bit here to ignore the num_discrete arg
        #  because cartpole is already discrete!
        num_discrete = len(actions)
        self.actions =  np.array(actions).reshape(
            num_discrete,
            *self.action_space_shape)
        return self.actions

    def sample_discrete_action(self):
        return random.choice(self.actions)


class PendulumEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('Pendulum-v0')
        super(PendulumEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = GlobalSpace([self.env.action_space])
        self.action_space_shape = self.action_space.shape

    def discretize_action_space(self, num_discrete):
        """
        Not every agent will need to do this
        """
        self.actions = np.linspace(self.action_space.low,
                                   self.action_space.high,
                                   num=num_discrete,
                                   endpoint=True)
        return self.actions

    def sample_discrete_action(self):
        return random.choice(self.actions)


class MountainCarEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('MountainCar-V0')
        super(MountainCarEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.obs_space_shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space_shape = (1,)

    def discretize_action_space(self, num_discrete):
        actions = [act for act in range(self.action_space.n)]

        #  cheat a bit here to ignore the num_discrete arg
        #  because mountain_car is already discrete!
        num_discrete = len(actions)
        self.actions =  np.array(actions).reshape(
            num_discrete,
            *self.action_space_shape)
        return self.actions

    def sample_discrete_action(self):
        return random.choice(self.actions)


env_register = {'Flex-v0': FlexEnvV0,
                'Flex-v1': FlexEnvV1,
                'Battery': BatteryEnv,
                'CartPole': CartPoleEnv,
                'Pendulum': PendulumEnv,
                'MountainCar': MountainCarEnv}


def make_env(env_id, **kwargs):
    logger.info('Making env {}'.format(env_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    env = env_register[env_id]

    return env(**kwargs)
