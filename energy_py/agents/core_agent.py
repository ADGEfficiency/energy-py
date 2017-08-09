"""
Module for Base_Agent & helper classes.
"""

import collections
import random

class Base_Agent(object):
    """
    the energy_py base agent class

    The methods of this class are:
        policy
        learning
    """

    def __init__(self, env, memory_length=int(1e6)):
        self.env = env
        self.memory_length = memory_length

        #  creating a memory object
        self.memory = Agent_Memory(memory_length=self.memory_length)

        #  grabbing the action space
        self.action_space = env.action_space

        return None

    #  assign errors for the Base_Agent methods
    def _reset(self): raise NotImplementedError
    def _act(self, observation): raise NotImplementedError
    def _learn(self, observation): raise NotImplementedError
    def _load_brain(self): raise NotImplementedError

    def reset(self):
        """
        """
        #  reset the memory
        self.memory.reset()

        return self._reset()

    def act(self, observation):
        """
        """
        return self._act(observation)

    def learn(self, memory):
        """
        """
        return self._learn()

    def load_brain(self, memory, ):
        """
        """
        return self._load_brain()


class Episode_History(object):
    """
    A class to hold the history of a single episode.
    """

    def __init__(self, episode_number):
        self.episode_number = episode_number
        self.experiences = []

    def add_to_history(self, observation,
                             action,
                             reward,
                             next_observation,
                             step):

        experience = (observation, action, reward, next_observation, step, self.episode_number)
        self.experiences.append(experience)


class Agent_Memory(object):
    """
    A class to hold the memory of an agent.
    """
    def __init__(self, memory_length):
        self.memory_length = memory_length
        self.reset()

    def reset(self):
        self.experiences = collections.deque([], self.memory_length)

    def add_episode(self, episode):
        self.experiences += episode.experiences

    def add_experience(self, experience):
        self.experiences +=experience

    def get_batch(self, batch_size):
        sample_size = min(batch_size, len(self.experiences))
        return random.sample(self.experiences, sample_size)


class Epsilon_Greedy(object):
    """
    A class to perform epsilon greedy action selection.

    Currently decay is done linearly.

    Decay occurs every time the object is used.
    """
    def __init__(self, decay_steps,
                       epsilon_start,
                       epsilon_end,
                       epsilon_test,
                       mode='learning'):

        #  we calculate a linear coefficient to decay with
        self.linear_coeff = (epsilon_end - epsilon_start) / decay_steps

        self.epsilon_test  = epsilon_test
        self.reset()

    def reset(self):
        """
        """
        self.steps         = 0
        self.epsilon       = self.epsilon_start

    def get_epsilon(self):
        """
        """
        if self.mode == 'testing':
            self.epsilon = self.epsilon_test

        elif self.steps < self.decay_steps:
            self.epsilon = self.linear_coeff * self.steps + self.start

        else:
            self.epsilon = self.epsilon_end

        return self.epsilon

    def select_action(self):
        return NotImplementedError
