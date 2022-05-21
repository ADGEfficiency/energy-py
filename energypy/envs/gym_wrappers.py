import energypy
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
            "mask": np.ones((1, self.env.observation_space.shape[0], self.env.observation_space.shape[0]))
        }, reward, done, {}

    def reset(self, mode='train'):
        if mode == 'test':
            self.n_test_eps -= 1
            if self.n_test_eps == 0:
                self.test_done = True

        return {
            "features": self.env.reset().reshape(1, *self.env.observation_space.shape),
            "mask": np.ones((1, self.env.observation_space.shape[0], self.env.observation_space.shape[0]))
        }

class ParallelGymObservationSpace:
    def __init__(self, envs):
        self.envs = envs
        self.shape = (len(envs), *envs[0].observation_space.shape)
        self.mask_shape = (len(envs), self.shape[1], self.shape[1])

    def sample(self):
        return [
            env.observation_space.sample()
            for env in self.envs
        ]


class ParallelGymActionSpace:
    def __init__(self, envs):
        self.envs = envs
        self.shape = (len(envs), *envs[0].action_space.shape)

        self.low = envs[0].action_space.low
        self.high = envs[0].action_space.high

    def sample(self):
        return np.array([env.action_space.sample() for env in self.envs])

    def contains(self, actions):
        check = []
        for env, act in zip(self.envs):
            assert env.action_space.contains(act)
        return True


class ParallelGymWrapper(energypy.envs.Base):
    def __init__(self, env_name, n_parallel=4, n_test_eps=16):

        self.n_test_eps = n_test_eps

        self.n_parallel = n_parallel
        #  could avoid the double env_name
        self.envs = [energypy.make(env_name, env_name) for _ in range(n_parallel)]
        env = self.envs[0]

        self.observation_space = ParallelGymObservationSpace(self.envs)
        self.action_space = ParallelGymActionSpace(self.envs)

        self.elements = (
            ('observation', (*self.observation_space.shape[1:], ), 'float32'),
            ('action', self.action_space.shape[1:], 'float32'),
            ('reward', (1, ), 'float32'),
            ('next_observation', self.observation_space.shape[1:], 'float32'),
            ('done', (1, ), 'bool'),
            ('observation_mask', self.observation_space.mask_shape[1:], 'float32'),
            ('next_observation_mask', self.observation_space.mask_shape[1:], 'float32'),
        )
        self.Transition = namedtuple('Transition', [el[0] for el in self.elements])

        #  in dev - will maybe replace elements with this
        self.elements_helper = {
            el[0]: el[1] for el in self.elements
        }

    def reset(self, mode='train'):
        if mode == 'test':
            self.n_test_eps -= 1
            if self.n_test_eps == 0:
                self.test_done = True

        #  just done for clarity
        features = [env.reset(mode) for env in self.envs]
        out = {
            'features': np.array([f['features'] for f in features]).reshape(self.observation_space.shape),
            'mask': np.array([f['mask'] for f in features]).reshape(self.observation_space.mask_shape)
        }
        return out

    def setup_test(self, n_test_eps):
        self.test_done = False
        self.n_test_eps = n_test_eps
        [env.setup_test(n_test_eps) for env in self.envs]

    def step(self, action):

        #  expect scaled action
        assert action.all() <= 1
        assert action.all() >= -1

        #  do the unscaling
        unscaled_action = action * self.action_space.high

        #  not sure why I need this ???
        # if 'lunar' in self.env_id.lower():
        #     unscaled_action = unscaled_action.reshape(action.shape[0], -1)

        from collections import defaultdict

        pkg = defaultdict(list)

        for env, act in zip(self.envs, unscaled_action):
            next_obs, reward, done, _ = env.step(act)

            pkg['next_observation'].append(next_obs['features'])
            pkg['next_observation_mask'].append(next_obs['mask'])
            pkg['reward'].append(reward)
            pkg['done'].append(done)

        #  dict of lists
        next_obs = {
            'features': pkg['next_observation'],
            'mask': pkg['next_observation_mask'],
        }
        next_obs['features'] = np.array(next_obs['features']).reshape(self.observation_space.shape)

        #  np arrays
        done = np.array(pkg['done']).reshape(-1, 1)
        reward = np.array(pkg['reward']).reshape(-1, 1)

        return next_obs, reward, done, {}

        # return {
        #     "features": next_obs.reshape(1, *self.env.observation_space.shape),
        #     "mask": np.ones((1, self.env.observation_space.shape[0], self.env.observation_space.shape[0]))
        # }, reward, done, {}
