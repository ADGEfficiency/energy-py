import numpy as np

from energy_py.agents import BaseAgent


class RandomAgent(BaseAgent):
    """
    An agent that samples the action space.

    Sample probability is used to control a balance between sampling and
    just doing nothing.

    args
        env (object) energy_py environment
        discount (float) discount rate
        sample_probability (float) sampling action space and doing nothing
    """
    def __init__(self,
                 env,
                 discount,
                 sample_probability=1.0):

        self.sample_probability = float(sample_probability)
        super().__init__(env, discount)

    def _act(self, **kwargs):
        """
        Agent selects action.

        returns
             action (np.array)
        """
        if np.random.rand() <= self.sample_probability:
            action = self.action_space.sample()
        else:
            action = self.action_space.low

        return action.reshape(1, self.action_space.shape[0])
