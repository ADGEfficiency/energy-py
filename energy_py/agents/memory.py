"""

"""

import collections
import itertools
import os

import numpy as np
import pandas as pd

from energy_py.main.scripts.utils import ensure_dir
from energy_py.main.scripts.visualizers import Agent_Memory_Visualizer


class Agent_Memory(Agent_Memory_Visualizer):
    """
    Inherits from Visualizer!

    Purpose of this class is to
        store the experiences of the agent
        process experiences for use by the agent to act or learn from

    The memory of the agent is two lists of experience numpy arrays
        self.experiences
        self.machine_experiences

    The two experience numpy arrays hold the following data
        experience = (observation, 0
                      action,
                      reward,
                      next_observation,
                      step,
                      episode) 5

        machine_experience = (observation, 0
                              action,
                              reward,
                              next_observation,
                              step,
                              episode,
                              discounted_return) 6

    experience = as observed

    machine experience = data for use in training neural networks
                       = data to learn from

    The machine_experience 'discounted_return' is a Monte Carlo return
    """

    def __init__(self, memory_length,
                       observation_space,
                       action_space,
                       reward_space,
                       discount_rate=0.99):

        super().__init__()
        self.memory_length = memory_length
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.discount_rate = discount_rate

        self.reset()

    def reset(self):
        """
        Resets the memory object
        """
        self.experiences = []
        self.machine_experiences = []
        self.losses = []

    def make_dummy_array(self, value, space):
        """
        Helper function
        Creates an array of dummy variables

        Not needed anymore
        """
        #  pull out the discrete_space space array
        discrete_space = space.discrete_space
        #  create an array of zeros
        scaled = np.zeros(discrete_space.shape)
        #  set to 1 where this value occurs
        scaled[np.where(discrete_space == value)] = 1
        #  quick check that we only have one dummy variable
        assert np.sum(scaled) == 1
        return scaled


    def scale_array(self, array, space):
        """
        Helper function for make_machine_experience()
        Uses the space & a given function to scale an array
        Default scaler is to normalize

        Used to scale the observation and action

        can probably move this into a parent
        """

        #  empty numpy array
        scaled_array = np.array([])

        #  iterate across the array values & corresponding space object
        for value, spc in itertools.zip_longest(array, space):
            if spc.type == 'continuous':
                # normalize continuous variables
                scaled = self.normalize(value, spc.low, spc.high)
            elif spc.type == 'discrete':
                #  shouldn't need to do anything
                #  check value is already dummy
                assert (value == 0) or (value == 1)
            else:
                assert 1 == 0

            #  appending the scaled value onto the scaled array
            scaled_array = np.append(scaled_array, scaled).reshape(-1)

        return scaled_array

    def make_machine_experience(self, exp, normalize_reward=True):
        """

        Helper function for add_experience
        Scales a given experience tuple

        Discounted return is an optimal arg so that the scaled_exp array can
        be created at any time

        """
        scaled_obs = self.scale_array(exp[0],
                                      self.observation_space)

        reward = exp[2]
        if normalize_reward:
            reward = self.normalize(exp[2],
                                    self.reward_space.low,
                                    self.reward_space.high)

        #  making an array for the scaled experience
        scaled_exp = np.array([scaled_obs,
                              exp[1],
                              reward,
                              None,
                              exp[4],
                              exp[5],
                              None])  # the Monte Carlo discounted return
        return scaled_exp

    def add_experience(self, observation, action, reward, next_observation, step, episode):
        """
        Adds a single step of experience to the two experiences lists
        """
        #  make the experience array
        exp = np.array([observation,
                       action,
                       reward,
                       next_observation,
                       step,
                       episode])

        #  make the machine experience array
        m_exp = self.make_machine_experience(exp, normalize_reward=False)
        #  add experiences to the memory
        self.experiences.append(exp)
        self.machine_experiences.append(m_exp)
        return None

    def finish_episode(self, episode_number):
        """
        perhaps this should occur in the agent?
        agent might want to do other stuff at end of episode
        """

        all_experiences = np.array(self.machine_experiences)
        assert all_experiences.shape[0] == len(self.machine_experiences)
        #  use boolean indexing to get experiences from last episode
        episode_mask = [all_experiences[:, 5] == episode_number]
        episode_experiences = all_experiences[episode_mask]

        #  now we can calculate the Monte Carlo discounted return
        R = 0
        returns, rewards = [], []
        for exp in episode_experiences[::-1]:
            r = exp[2]
            R = r + self.discount_rate * R  # the Bellman equation
            returns.insert(0, R)
            rewards.append(r)

        total_reward = sum(rewards)
        max_reward = max(rewards)

        #  now we normalize the episode returns
        rtns = np.array(returns)
        rtns = (rtns - rtns.mean()) / (rtns.std())

        #  now we have the normalized episode returns
        #  we can fill in the returns each experience in machine_experience
        #  (for this episode)

        new_exps = []
        for exp, rtn in itertools.zip_longest(episode_experiences, rtns):
            exp[6] = rtn
            new_exps.append(exp)

        idx_array = np.arange(all_experiences.shape[0])
        assert idx_array.shape[0] == all_experiences.shape[0]
        episode_indicies = idx_array[episode_mask]
        start = episode_indicies[0]
        end = episode_indicies[-1] + 1
        self.machine_experiences[start:end] = new_exps

        print('Ep {} total reward is {:.2f} max reward is {:.2f}'.format(episode_number,
                                                        total_reward,
                                                        max_reward))
        return None

    def get_episode_batch(self, episode_number):
        """
        Gets the experiences for a given episode.

        Quite inefficient as we loop over the entire scaled experiences list.
        """
        scl_episode_experiences = []

        for scl_exp in self.machine_experiences:
            if scl_exp[5] == episode_number:
                scl_episode_experiences.append(scl_exp)
                assert scl_exp[5] == episode_number

        observations = np.array(
            [exp[0] for exp in scl_episode_experiences]).reshape(-1, len(self.observation_space))
        actions = np.array(
            [exp[1] for exp in scl_episode_experiences]).reshape(-1, len(self.action_space))
        returns = np.array(
            [exp[6] for exp in scl_episode_experiences]).reshape(-1, 1)

        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == returns.shape[0]

        return observations, actions, returns

    def get_random_batch(self, batch_size):
        """
        Gets a random batch of experiences.

        can def do this better - and maybe share code with get_episode_batch
        """
        sample_size = min(batch_size, len(self.machine_experiences))

        #  limiting the scaled_experiences list to the memory length
        memory = self.experiences[-self.memory_length:]
        scaled_memory = self.machine_experiences[-self.memory_length:]

        assert len(memory) == len(scaled_memory
                                  )
        #  indicies for the batch
        indicies = np.random.randint(low=0,
                                     high=len(memory),
                                     size=sample_size)

        #  randomly sample from the memory & returns
        memory_batch = [memory[i] for i in indicies]
        scaled_memory_batch = [scaled_memory[i] for i in indicies]

        obs = [exp[0] for exp in scaled_memory_batch]
        acts = [exp[1] for exp in memory_batch]
        rtns = [exp[6] for exp in scaled_memory_batch]

        observations = np.array(obs).reshape(-1, len(self.observation_space))
        actions = np.array(acts).reshape(-1, len(self.action_space))
        returns = np.array(rtns).reshape(-1, 1)

        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == returns.shape[0]

        return observations, actions, returns
