"""
A registery for environments supported by energy_py

Combination of native energy_py environments and wrapped gym environments
"""

import logging
import random

import gym
import numpy as np

from energy_py.common import GlobalSpace

from energy_py.envs.flex import Flex

from energy_py.envs.battery.battery import Battery


logger = logging.getLogger(__name__)


class EnvWrapper(object):

    def __init__(self, env):
        self.env = env

    def __repr__(self):
        return repr(self.env)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def seed(self, seed=None):
        if seed:
            return self.env.seed(int(seed))


class CartPoleEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('CartPole-v0')
        super(CartPoleEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.observation_space.shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space.shape = (1,)
        self.action_space.discretize = self.discretize_action_space
        self.action_space.sample_discrete = self.sample_discrete_action

    def step(self, action):
        #  cartpole doesn't accept an array!
        return self.env.step(action[0][0])

    def discretize_action_space(self, num_discrete):
        actions = [act for act in range(self.action_space.n)]

        num_discrete = len(actions)
        self.actions =  np.array(actions).reshape(
            num_discrete,
            *self.action_space.shape)

        return self.actions

    def sample_discrete_action(self):
        return random.choice(self.actions)


class PendulumEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('Pendulum-v0')
        super(PendulumEnv, self).__init__(env)

        self.observation_space = self.env.observation_space
        self.observation_space.shape = self.observation_space.shape

        self.action_space = GlobalSpace([self.env.action_space])
        self.action_space.shape = self.action_space.shape
        self.action_space.discretize = self.discretize_action_space
        self.action_space.sample_discrete = self.sample_discrete_action

    def discretize_action_space(self, num_discrete):
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
        self.observation_space.shape = self.observation_space.shape

        self.action_space = self.env.action_space
        self.action_space.shape = (1,)
        self.action_space.discretize = self.discretize_action_space
        self.action_space.sample_discrete = self.sample_discrete_action

    def discretize_action_space(self, num_discrete):
        actions = [act for act in range(self.action_space.n)]

        num_discrete = len(actions)
        self.actions =  np.array(actions).reshape(
            num_discrete,
            *self.action_space.shape)
        return self.actions

    def sample_discrete_action(self):
        return random.choice(self.actions)


env_register = {'flex': Flex,
                'battery': Battery,
                'cartpole-v0': CartPoleEnv,
                'pendulum-v0': PendulumEnv,
                'mountaincar-v0': MountainCarEnv}


def make_env(env_id, **kwargs):
    logger.info('Making env {}'.format(env_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    env = env_register[env_id]

    return env(**kwargs)
