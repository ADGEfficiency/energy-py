from collections import deque
import numpy as np


class RandomPolicy():
    def __init__(self, env):
        self.env = env

    def __call__(self, observation=None):
        unscaled = self.env.action_space.sample().reshape(-1, *self.env.action_space.shape)
        scaled = unscaled / abs(self.env.action_space.high)
        return scaled, None, None


class FixedPolicy():
    def __init__(self, env, actions):
        self.env = env
        self.actions = deque(actions)

    def __call__(self, observation=None):
        action = self.actions.popleft()
        action = np.array(action).reshape(-1, *self.env.action_space.shape)
        return action, None, None


def make(env):
    return RandomPolicy(env)
