""" Naive agents used as baselines """
import numpy as np

from energypy.agents.agent import BaseAgent


class RandomAgent(BaseAgent):
    """ randomly samples action space """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _act(self, observation, **kwargs):
        return self.action_space.sample()

    def _learn(self, *args, **kwargs):
        pass


class NoOp(BaseAgent):
    """ does nothing each step """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _act(self, observation, **kwargs):
        return self.action_space.no_op

    def _learn(self, *args, **kwargs):
        pass
