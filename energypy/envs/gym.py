from collections import defaultdict

import gym

from energypy.common.spaces import ActionSpace
from energypy.common.spaces import PrimitiveConfig as Prim


class EnvWrapper(object):

    def __init__(self, env):
        self.env = env
        self.info = defaultdict(list)

    def __repr__(self):
        return repr(self.env)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def seed(self, seed=None):
        if seed:
            return self.env.seed(int(seed))

        #  should just inherit from the gym envs - TODO
        #  from gym.envs.classic_control import CartPoleEnv

class CartPoleEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('CartPole-v0')
        super(CartPoleEnv, self).__init__(env)

        self.observation_space = self.env.observation_space

        self.action_space = ActionSpace('action').from_primitives(
            Prim('left_or_right', 0, 2, 'discrete', None)
        )

    def step(self, action):
        #  doesn't accept an array!
        next_state, reward, done, info = self.env.step(action[0][0])
        self.info['action'].append(action[0][0])
        self.info['reward'].append(reward)
        return next_state, reward, done, self.info


class PendulumEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('Pendulum-v0')
        super(PendulumEnv, self).__init__(env)

        self.observation_space = self.env.observation_space

        self.action_space = ActionSpace('action').from_primitives(
            Prim('torque', -env.env.max_torque, env.env.max_torque, 'continuous', None)
        )


class MountainCarEnv(EnvWrapper):

    def __init__(self):
        env = gym.make('MountainCar-v0')
        super(MountainCarEnv, self).__init__(env)

        self.observation_space = self.env.observation_space

        self.action_space = ActionSpace('action').from_primitives(
            Prim('left_or_right', 0, 2, 'discrete', None)
        )

    def step(self, action):
        #  doesn't accept an array!
        return self.env.step(action[0][0])
