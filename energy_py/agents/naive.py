import numpy as np

from energy_py.agents import BaseAgent


class NaiveBatteryAgent(BaseAgent):
    """
    This naive agent takes actions used predefined rules.

    A naive agent is useful as a baseline for comparing with reinforcement
    learning agents.

    As the rules are predefined each agent is specific to an environment.

    This agent is designed to control the battery environment.
    """

    def __init__(self, env, discount):
        #  calling init method of the parent Base_Agent class
        super().__init__(env, discount)

    def _reset(self):
        #  nothing additional to be reset for this agent
        return None

    def _act(self, **kwargs):
        """
        Agent recieves a numpy array as the observation

        Agent makes determinstic actions based on a timestamp

        Doesn't look at the observation numpy array that other
        agents use (it is available in kwargs)
        """
        hour_index = self.observation_info.index('D_hour')
        observation = kwargs['observation']
        hour = observation[0][hour_index]

        #  grab the spaces list
        act_spaces = self.action_space.spaces

        if hour >= 7 and hour < 10:
            #  discharge during morning peak
            action = [act_spaces[0].low, act_spaces[1].high]

        elif hour >= 15 and hour < 21:
            #  discharge during evening peak
            action = [act_spaces[0].low, act_spaces[1].high]

        else:
            #  charge at max rate
            action = [act_spaces[0].high, act_spaces[1].low]

        return np.array(action).reshape(1, self.action_space.shape[0])


class DispatchAgent(BaseAgent):

    def __init__(self, env, discount, trigger=200):
        #  calling init method of the parent Base_Agent class
        super().__init__(env, discount)
        self.trigger = trigger

    def _act(self, **kwargs):

        obs = kwargs['observation']
        idx = self.env.observation_info.index('C_cumulative_mean_dispatch_[$/MWh]')
        cumulative_dispatch = obs[0][idx]

        if cumulative_dispatch > 200:
            action = self.action_space.high

        else:
            action = self.action_space.low

        return np.array(action).reshape(1, self.action_space.shape[0])

    def _reset(self):
        #  nothing additional to be reset for this agent
        return None
