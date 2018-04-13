from collections import namedtuple
import operator

import numpy as np

from energy_py.agents import BaseAgent

ClassifierCondition = namedtuple('Condition', ['horizion', 'bin', 'operation'])


class ClassifierStragety(object):
    """
    A set of rules for taking deterministic actions based on an observation

    args
        stragety (ClassifierStragety) contains the conditions
        action (np.array) taken if all the stragety is true
        observation_info (list) list of strings for each dim of the observation

    The conditions list
    """

    def __init__(self, conditions, action, observation_info):
        self.conditions = conditions
        self.action = action
        self.observation_info = observation_info

        #  a dictionary for operators to allow different in/equalities
        #  to be used
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

        Creates the string of the observation
        Predicition indexed using this string
        Returns operation of the prediction versus the condition

        Either
            prediction == 1
            prediction != 1
        """
        string = 'D_h_{}_Predicted_Price_Bin_{}'.format(condition.horizion,
                                                        condition.bin)

        #  get the index and the prediction
        idx = self.observation_info.index(string)
        prediciton = observation[0, idx]

        #  get the operation function from self.operators dict
        operation = self.operators[condition.operation]

        #  compare the prediciton verus 1
        return operation(prediciton, 1)

    def check_observation(self, observation):
        """
        Checks the observation versus all conditions

        args
            observation (np.array)

        returns
            action (np.array)

        Iterates over all the conditions in this ClassifierStragety
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

    def __init__(self,
                 stragety,
                 conditions,
                 action,
                 obs_info, **kwargs):
        super().__init__(**kwargs)

        #  hack to get around env adding
        new_obs_info = self.env.obs_spaces[len(obs_info):]
        strat_obs_info = obs_info.extend(new_obs_info)

        self.stragety = ClassifierStragety(conditions,
                                           action,
                                           strat_obs_info)

    def _act(self, observation):
        return self.stragety.check_observation(observation)
