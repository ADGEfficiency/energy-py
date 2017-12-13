import numpy as np

from energy_py.agents import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self,
                 env,
                 discount,
                 brain_path=[]):

        super().__init__(env, discount, brain_path)

    def _reset(self):
        pass

    def _act(self, **kwargs):
        action = self.action_space.sample()
        action = np.array(action).reshape(1, self.action_space.shape[0])
        return action

