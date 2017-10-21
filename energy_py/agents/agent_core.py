"""
Module for Base_Agent & helper classes.
"""

import numpy as np

from energy_py.agents import Agent_Memory
from energy_py.main.scripts.utils import Utils

class Base_Agent(Utils):
    """
    the energy_py base agent class

    The main methods of this class are:
        act
        learn
    """

    def __init__(self, env, discount, epsilon_decay_steps=0,
                 memory_length=int(1e6), verbose=False):

        super().__init__(verbose)

        self.env = env
        self.discount = discount
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_actions = len(self.action_space)
        self.observation_dim = len(self.observation_space)

        self.memory_length = memory_length

        #  object to hold all of the agents experience
        self.memory = Agent_Memory(memory_length=self.memory_length,
                                   observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   reward_space=env.reward_space,
                                   discount=discount,
                                   verbose=self.verbose)

        return None

    #  assign errors for the Base_Agent methods
    def _reset(self): raise NotImplementedError
    def _act(self, observation): raise NotImplementedError
    def _learn(self, observation): raise NotImplementedError
    def _load_brain(self): raise NotImplementedError
    def _save_brain(self): raise NotImplementedError
    def _output_results(self): raise NotImplementedError

    def reset(self):
        """
        """
        #  reset the objects set in the Base_Agent init
        self.memory.reset()
        self.e_greedy.reset()

        return self._reset()

    def act(self, **kwargs):
        """
        Main function for agent to take action.

        Calls the ._act method (which can be overidden in the agent child class)
        """

        return self._act(**kwargs)

    def learn(self, **kwargs):
        """
        """
        return self._learn(**kwargs)

    def load_brain(self):
        """
        """
        return self._load_brain()

    def save_brain(self):
        """
        """
        return self._save_brain()

    def output_results(self):
        """
        """
        return self.memory.output_results()

    def all_state_actions(self, action_space, observation):
        """
        All possible combinations actions for a single observation

        Used by Q-Learning for both acting and learning
            acting = argmax Q(s,a) for all possible a to select action
            learning = argmax Q(s',a) for all possible a to create Bellman target

        action_combinations = act_dim[0] * act_dim[1] ... * act_dim[n]
                              (across the action_space)

        args
            action_space    : a list of Space objects
            observation     : np array (1, observation_dim)

        returns
            state_acts      : np array (action_combinations,
                                        observation_dim + num_actions)
            actions         : np array (action_combinations,
                                        num_actions)
        """
        self.verbose_print('creating state action combinations')
        #  get the discrete action space for all action dimensions
        #  list is used to we can use itertools.product below
        disc_action_spaces = [list(space.discretize()) for space in action_space]

        #  create every possible combination of actions
        #  this creates the unscaled actions
        actions = [act for act in itertools.product(*disc_action_spaces)]

        #  scale the actions
        scaled_actions = [scale_array(act, action_space, normalize) for act in actions]

        #  create an array with one obs per possible action combinations
        #  reshape into (num_actions, observation_dim)
        observations = np.tile(observation, actions.shape[0])
        observations = observations.reshape(actions.shape[0], observation.shape[1])

        #  concat the observations & actions
        #  used scaled actions
        state_acts = np.concatenate([observations, scaled_actions], axis=1)
        assert actions.shape[0] == state_acts.shape[0]
        return state_acts, actions



class Epsilon_Greedy(object):
    """
    A class to perform epsilon greedy action selection.

    Currently decay is done linearly.

    Decay occurs every time the object is used.
    """
    def __init__(self, decay_steps,
                       epsilon_start = 1.0,
                       epsilon_end   = 0.1,
                       epsilon_test  = 0.0,  #  default not at zero to test generalization
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
            print('epsilon {:.3f}'.format(self.epsilon))

        if self.mode == 'testing':
            self.epsilon = self.epsilon_test

        elif self.steps < self.decay_steps:
            self.epsilon = self.linear_coeff * self.steps + self.epsilon_start

        else:
            self.epsilon = self.epsilon_end

        self.steps += 1

        return float(self.epsilon)
