import numpy as np

from .game_2048 import Games, empty_boards

from energypy.envs import BaseEnv

from energypy.common.spaces import DiscreteSpace, GlobalSpace

from energypy.common.spaces.continuous import ImageSpace

class Game2048(object):

    def __init__(
            self,
            observation_dims='flat'
    ):
        self.n_boards = 1
        self.N = 4
        self.game = Games(
            n_boards=self.n_boards,
            N=self.N,
            observation_dims=observation_dims
        )

        self.action_space = GlobalSpace('action').from_spaces(
            DiscreteSpace(4),
            'move_tile'
        )

        if observation_dims == 'flat':
            self.observation_space = ImageSpace(
                low=0, high=2048, shape=(16,)
            )

        elif observation_dims == '2D':
            self.observation_space = ImageSpace(
                low=0, high=2048, shape=(4, 4, 1)
            )

    def step(self, action):
        action = action.reshape(self.n_boards)
        return self.game.step(action)

    def reset(self):
        self.game.boards = empty_boards(
            self.n_boards, self.N
        )
        return self.game.boards
