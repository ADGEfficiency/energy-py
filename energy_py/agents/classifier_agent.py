from collections import namedtuple
import operator

import numpy as np

from energy_py.agents import BaseAgent

ClassifierCondition = namedtuple('Condition', ['horizion', 'bin', 'operation'])


class ClassifierStragety(object):
    """
    Implements a set of rules for taking deterministic actions based on
    an observation

    args
        conditions (list) a list of conditions, action taken if all are true
        action (np.array) the action taken if all conditions are true
        observation_info (list) list of strings for each dim of the observation
    """

    def __init__(self, conditions, action, observation_info):
        self.conditions = conditions
        self.action = action
        self.observation_info = observation_info

        self.operators = {'==': operator.eq,
                          '!=': operator.ne}

    def compare_condition(self, observation, condition):
        """
        Checks if a condition is true

        args
            observation (np.array)
            condition (namedtuple)

        returns
            (bool)
        """
        string = 'D_h_{}_Predicted_Price_Bin_{}'.format(condition.horizion,
                                                        condition.bin)

        idx = self.observation_info.index(string)

        prediciton = observation[0, idx]

        operation = self.operators[condition.operation]

        return operation(prediciton, 1)

    def check_observation(self, observation):
        """
        Checks the observation versus all conditions

        args
            observation (np.array)

        returns
            action (np.array)
        """
        bools = [self.compare_condition(observation, c)
                 for c in self.conditions]

        if all(bools):
            return self.action
        else:
            return np.zeros_like(self.action)


class ClassifierAgent(BaseAgent):
    """
    Flexes based on a prediction from a classifier.
    """

    def __init__(self, env, discount, strageties, **kwargs):

        super().__init__(env, discount, memory_length=10)

        self.strageties = strageties

    def _act(self, observation):
        pass
