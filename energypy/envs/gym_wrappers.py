from collections import namedtuple

import numpy as np
import gym

from energypy.envs.base import AbstractEnv


#  key=name, value=id
env_ids = {
    'pendulum': 'Pendulum-v1',
    'lunar': 'LunarLanderContinuous-v2'
}


def inverse_scale(action, low, high):
    return action * (high - low) + low


class GymWrapper(AbstractEnv):
    def __init__(self, env_name):
        self.env_id = env_ids[env_name]
        self.env = gym.make(self.env_id)
        self.elements = (
            ('observation', self.env.observation_space.shape, 'float32'),
            ('action', self.env.action_space.shape, 'float32'),
            ('reward', (1, ), 'float32'),
            ('next_observation', self.env.observation_space.shape, 'float32'),
            ('done', (1, ), 'bool'),
            ('observation_mask', self.env.observation_space.shape, 'float32'),
            ('next_observation_mask', self.env.observation_space.shape, 'float32'),
        )
        self.Transition = namedtuple('Transition', [el[0] for el in self.elements])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def setup_test(self, n_test_eps):
        self.test_done = False
        self.n_test_eps = n_test_eps

    def step(self, action):
        #  expect a scaled action here
        assert action.all() <= 1
        assert action.all() >= -1
        unscaled_action = action * self.env.action_space.high
        if 'lunar' in self.env_id.lower():
            unscaled_action = unscaled_action.reshape(-1)
        next_obs, reward, done, _ = self.env.step(unscaled_action)
        return {
            "features": next_obs.reshape(1, *self.env.observation_space.shape),
            "mask": np.ones((1, *self.env.observation_space.shape))
        }, reward, done, {}

    def reset(self, mode='train'):
        if mode == 'test':
            self.n_test_eps -= 1
            if self.n_test_eps == 0:
                self.test_done = True

        return {
            "features": self.env.reset().reshape(1, *self.env.observation_space.shape),
            "mask": np.ones((1, *self.env.observation_space.shape))
        }
