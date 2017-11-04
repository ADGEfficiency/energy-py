import itertools

import numpy as np

from energy_py.agents.memory import Agent_Memory
from energy_py.main.scripts.utils import Utils

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

    def __init__(self, env, discount, verbose):
        #  send up verbose up to Utils class
        super().__init__(verbose)

        self.env = env
        self.discount = discount

        #  use the env to setup the agent
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_actions = len(self.action_space)
        self.observation_dim = len(self.observation_space)

        #  create a memory for the agent
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
        Resets the agent
        """
        #  reset the objects set in the Base_Agent init
        self.memory.reset()
        return self._reset()

    def act(self, observation):
        """
        Action selection by agent

        args
            observation (np array) : shape=(1, observation_dim)

        return
            action (np array) : shape=(1, num_actions)
        """
        self.verbose_print('acting')
        return self._act(observation)

    def learn(self, **kwargs):
        """
        Agent learns from experience

        Use **kwargs for flexibility

        return
            training_history (object) : info about learning (i.e. loss)
        """
        self.verbose_print('learning')
        return self._learn(**kwargs)

    def load_brain(self):
        """
        Agent can load previously created memories, policies or value functions
        """
        return self._load_brain()

    def save_brain(self):
        """
        Agent can save previously created memories, policies or value functions
        """
        return self._save_brain()

    def output_results(self):
        """
        Save results from the agents memory
        """
        return self.memory.output_results()

    def all_state_actions(self, action_space, observation):
        """
        This is a helper function used by value function based agents

        All possible combinations actions for a single observation

        Used by Q-Learning for both acting and learning
            acting = argmax Q(s,a) for all possible a to select action
            learning = argmax Q(s',a) for all possible a to create Bellman target

        action_combinations = act_dim[0] * act_dim[1] ... * act_dim[n]
                              (across the action_space)

        args
            action_space    : a list of Space objects
            observation     : np array (1, observation_dim)
                              should be already scaled

        returns
            state_acts      : np array (action_combinations,
                                        observation_dim + num_actions)
            actions         : np array (action_combinations,
                                        num_actions)
        """
        #  get the discrete action space for all action dimensions
        #  list is used to we can use itertools.product below
        disc_action_spaces = [space.discretize(length=20) for space in action_space]

        #  create every possible combination of actions
        #  this creates the unscaled actions
        actions = np.array([act for act in itertools.product(*disc_action_spaces)])

        #  scale the actions
        scaled_actions = np.array([self.scale_array(act, action_space) for act
                                   in actions]).reshape(actions.shape)

        #  create an array with one obs per possible action combinations
        #  reshape into (num_actions, observation_dim)
        observations = np.tile(observation, actions.shape[0])
        observations = observations.reshape(actions.shape[0], self.observation_dim) 

        #  concat the observations & actions
        state_acts = np.concatenate([observations, scaled_actions], axis=1)
        assert actions.shape[0] == state_acts.shape[0]

        return state_acts, actions


class Epsilon_Greedy(object):
    """
    A class to perform epsilon greedy action selection.

    Decay is done linearly.

    Decay occurs every time we call get_epsilon.
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
        Reset the Epsilon_Greedy object
        """
        self.steps   = 0
        self.epsilon = self.epsilon_start
        self.mode    = 'training'

    def get_epsilon(self):
        """
        Get the current value of epsilon

        returns
            epsilon (float)
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
