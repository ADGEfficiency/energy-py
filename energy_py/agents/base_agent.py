import itertools

import numpy as np

from energy_py.agents.memory import Agent_Memory
from energy_py import Utils

class Base_Agent(Utils):
    """
    The energy_py base agent class

    The main methods of this class are
        reset
        act
        learn

    All agents should override the following methods
        _reset
        _act
        _learn

    Some agents will also override
        _load_brain
        _save_brain
        _output_results

    args
        env      : energy_py environment
        discount : float : discount rate (gamma)

    methods
        all_state_actions : used to create all combinations of state across the
                            action space
    """

    def __init__(self, memory_length=10000, **kwargs):
        #  send up verbose up to Utils class
        verbose = kwargs.pop('verbose')
        super().__init__(verbose)

        self.env = kwargs.pop('env')
        self.discount = kwargs.pop('discount')
        self.brain_path = kwargs.pop('brain_path')

        #  use the env to setup the agent
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_space = self.env.reward_space
        self.num_actions = len(self.action_space)
        self.observation_dim = len(self.observation_space)

        #  create a memory for the agent
        #  object to hold all of the agents experience
        self.memory = Agent_Memory(observation_space=self.observation_space,
                                   action_space=self.action_space,
                                   reward_space=self.reward_space,
                                   discount=self.discount,
                                   memory_length=memory_length,
                                   verbose=verbose)

    #  assign errors for the Base_Agent methods
    def _reset(self): raise NotImplementedError
    def _act(self, observation): raise NotImplementedError
    def _learn(self, observation): raise NotImplementedError
    def _load_brain(self): raise NotImplementedError
    def _save_brain(self): raise NotImplementedError
    def _output_results(self): raise NotImplementedError

    def reset(self):
        """
        Resets the agent
        """
        #  reset the objects set in the Base_Agent init
        self.memory.reset()
        return self._reset()

    def act(self, **kwargs):
        """
        Action selection by agent

        args
            observation (np array) : shape=(1, observation_dim)

        return
            action (np array) : shape=(1, num_actions)
        """
        self.verbose_print('Agent is acting', level=2)
        return self._act(**kwargs)

    def learn(self, **kwargs):
        """
        Agent learns from experience

        Use **kwargs for flexibility

        return
            training_history (object) : info about learning (i.e. loss)
        """
        self.verbose_print('Agent is learning', level=1)
        return self._learn(**kwargs)

    def load_brain(self):
        """
        Agent can load previously created memories, policies or value functions
        """
        self.verbose_print('Loading agent brain', level=0)
        return self._load_brain()

    def save_brain(self):
        """
        Agent can save previously created memories, policies or value functions
        """
        self.verbose_print('Saving agent brain', level=0)
        return self._save_brain()

    def output_results(self):
        """
        Agent can load previously created memories, policies or value functions
        """
        return self.memory.output_results()

    def setup_all_state_actions(self, spc_len):
        """
        Creating the combination of all actions with a single obervation is
        one of the reasons value function methods are expensive.

        This function is designed to run once on init, doing the things that
        only need to be done once for state_action creation.

        args
            spc_len (int) : the length of the discrteizied action space

        returns
            scaled_actions (np.array) : an array of the scaled actions
                                        shape=(spc_len, num_actions)
        """
        self.verbose_print('Setting up self.scaled_actions')

        #  get the discrete action space for all action dimensions
        #  list is used to we can use itertools.product below
        disc_action_spaces = [space.discretize(length=spc_len) for space in self.action_space]
        self.verbose_print('discrete action space is {}'.format(disc_action_spaces))

        #  create every possible combination of actions
        #  this creates the unscaled actions
        self.actions = np.array([act for act in itertools.product(*disc_action_spaces)])

        #  scale the actions
        scaled_actions = np.array([self.scale_array(act, self.action_space) for act
                                   in self.actions]).reshape(self.actions.shape)

        self.verbose_print('scaled_actions shape is {}'.format(scaled_actions.shape))
        self.verbose_print('scaled_actions are {}'.format(scaled_actions))
        assert self.actions.shape[0] == scaled_actions.shape[0]
        return scaled_actions

    def all_state_actions(self, observation):
        """
        This is a helper function used by value function based agents

        All possible combinations actions for a single observation

        Used by Q-Learning for both acting and learning
            acting = argmax Q(s,a) for all possible a to select action
            learning = argmax Q(s',a) for all possible a to create Bellman target

        action_combinations = act_dim[0] * act_dim[1] ... * act_dim[n]
                              (across the action_space)

        Note that the method setup_all_state_actions should be run prior to
        this function (should be run during child agent __init__)

        args
            observation     : np array (1, observation_dim)
                              should be already scaled

        returns
            state_acts      : np array (action_combinations,
                                        observation_dim + num_actions)
            self.actions    : np array (action_combinations,
                                        num_actions)
        """
        #  create an array with one obs per possible action combinations
        #  reshape into (num_actions, observation_dim)
        observations = np.tile(observation, self.scaled_actions.shape[0])
        observations = observations.reshape(self.scaled_actions.shape[0], self.observation_dim)

        #  concat the observations & actions
        state_acts = np.concatenate([observations, self.scaled_actions], axis=1)

        assert state_acts.shape[0] == self.scaled_actions.shape[0]

        return state_acts, self.actions


class Epsilon_Greedy(object):
    """
    A class to perform epsilon greedy action selection.

    Decay is done linearly.

    Decay occurs every time we call get_epsilon.
    """
    def __init__(self, decay_steps,
                       epsilon_start = 1.0,
                       epsilon_end   = 0.1):

        #  we calculate a linear coefficient to decay with
        self.linear_coeff = (epsilon_end - epsilon_start) / decay_steps

        self.decay_steps   = decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end

        self.reset()

    def reset(self):
        self.steps = 0
        self._epsilon = self.epsilon_start
        print('starting epsilon at {}'.format(self.epsilon_start))

    @property
    def epsilon(self):
        if self.steps < self.decay_steps:
            self._epsilon = self.linear_coeff * self.steps + self.epsilon_start
        else:
            self._epsilon = self.epsilon_end

        self.steps += 1
        return float(self._epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
