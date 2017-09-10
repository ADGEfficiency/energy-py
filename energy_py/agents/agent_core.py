"""
Module for Base_Agent & helper classes.
"""

import numpy as np

from energy_py.agents.memory import Agent_Memory

class Base_Agent(object):
    """
    the energy_py base agent class

    The main methods of this class are:
        act
        learn
    """

    def __init__(self, env, epsilon_decay_steps=10000, memory_length=int(1e6),
                 discount_rate=0.95, verbose=0):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_actions = len(self.action_space)
        self.observation_dim = len(self.observation_space)

        self.memory_length = memory_length
        self.discount_rate = discount_rate
        self.epsilon_decay_steps = epsilon_decay_steps
        self.verbose = verbose

        #  object to use to decay epsilon for action selection
        self.epsilon_greedy = Epsilon_Greedy(decay_steps=self.epsilon_decay_steps,
                                             verbose=0)

        #  object to hold all of the agents experience
        self.memory = Agent_Memory(memory_length=self.memory_length,
                                   observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   reward_space=env.reward_space,
                                   discount_rate=discount_rate)

        return None

    #  assign errors for the Base_Agent methods
    def _reset(self): raise NotImplementedError
    def _act(self, observation): raise NotImplementedError
    def _learn(self, observation): raise NotImplementedError
    def _load_brain(self): raise NotImplementedError
    def _output_results(self): raise NotImplementedError

    def reset(self):
        """
        """
        #  reset the objects set in the Base_Agent init
        self.memory.reset()
        self.epsilon_greedy.reset()

        return self._reset()

    def act(self, observation,
                  session = None,
                  epsilon = None):
        """
        Main function for agent to take action.

        Calls the ._act method (which can be overidden in the agent child class)
        """
        epsilon = self.epsilon_greedy.get_epsilon()

        return self._act(observation, session, epsilon)

    def learn(self, observations       = None,
                    actions            = None,
                    discounted_returns = None,
                    session            = None):
        """
        """
        assert not np.any(np.isnan(observations))
        assert not np.any(np.isnan(actions))
        assert not np.any(np.isnan(discounted_returns))
        print('epsilon is {}'.format(self.epsilon_greedy.epsilon))
        if self.verbose > 0:
            print('Learning')
            print('observations are {}'.format(observations))
            print('actions are {}'.format(actions))
            print('discounted_returns are {}'.format(discounted_returns))

        return self._learn(observations, actions, discounted_returns, session)

    def load_brain(self):
        """
        """
        return self._load_brain()

    def output_results(self):
        """
        Keeping this simple for now
        """
        self.memory.output_results
        return self.memory.outputs


class Epsilon_Greedy(object):
    """
    A class to perform epsilon greedy action selection.

    Currently decay is done linearly.

    Decay occurs every time the object is used.
    """
    def __init__(self, decay_steps,
                       epsilon_start = 1.0,
                       epsilon_end   = 0.1,
                       epsilon_test  = 0.05,  #  default not at zero to test generalization
                       mode          = 'learning',
                       verbose       = 0):

        #  we calculate a linear coefficient to decay with
        self.linear_coeff = (epsilon_end - epsilon_start) / decay_steps

        self.decay_steps   = decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_test  = epsilon_test
        self.verbose       = verbose

        self.reset()

    def reset(self):
        """
        """
        self.steps   = 0
        self.epsilon = self.epsilon_start
        self.mode    = 'training'

    def get_epsilon(self):
        """
        """

        if self.verbose:
            print('mode is {}'.format(self.mode))
            print('steps taken {}'.format(self.steps))
            print('epsilon {}'.format(self.epsilon))

        if self.mode == 'testing':
            self.epsilon = self.epsilon_test

        elif self.steps < self.decay_steps:
            self.epsilon = self.linear_coeff * self.steps + self.epsilon_start

        else:
            self.epsilon = self.epsilon_end

        self.steps += 1


        return self.epsilon
