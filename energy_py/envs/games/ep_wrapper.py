import numpy as np

from .game_2048 import Games, empty_boards

from energy_py.envs import BaseEnv

from energy_py.common.spaces import DiscreteSpace, GlobalSpace

from energy_py.common.spaces.continuous import ImageSpace

class Game2048(object):

    def __init__(self):
        self.n_boards = 1
        self.N = 4
        self.game = Games(
            n_boards=self.n_boards,
            N=self.N
        )

        self.action_space = GlobalSpace('action').from_spaces(
            DiscreteSpace(4),
            'move_tile'
        )

        self.observation_space = ImageSpace(low=0, high=2048, shape=(16,))

    def step(self, action):
        action = action.reshape(self.n_boards)
        return self.game.step(action)

    def reset(self):
        self.game.boards = empty_boards(
            self.n_boards, self.N
        )
        return self.game.boards

env = Game2048()

obs = env.reset()

obs, rew, done, _ = env.step(np.array([1]))
