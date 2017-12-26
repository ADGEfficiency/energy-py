import pandas as pd
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
        #  passing the environment to the Base_Agent
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
            action = np.array(action).reshape(1, self.action_space.shape[0])

        elif hour >= 15 and hour < 21:
            #  discharge during evening peak
            action = [act_spaces[0].low, act_spaces[1].high]
            action = np.array(action).reshape(1, self.action_space.shape[0])

        else:
            #  charge at max rate
            action = [act_spaces[0].high, act_spaces[1].low]
            action = np.array(action).reshape(1, self.action_space.shape[0])

        print('hour {}'.format(hour))
        print('action {}'.format(action))
        return np.array(action).reshape(-1, self.action_space.shape[0])

    def _learn(self):
        print('I am an agent based on a human desgined heuristic')
        print('I cannot learn anything')
        return None

    def _load_brain(self):
        print('I am an agent based on a human desgined heuristic')
        print('I have no brain')
        #  TODO could get this to load a rule from disk
        return None
