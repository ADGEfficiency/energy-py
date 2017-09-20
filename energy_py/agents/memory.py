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
    inherits from Visualizer!

    A class to hold the memory of an agent

    Contains functions to process the memory for an agent to learn from

    two numpy arrays to hold experience

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

    machine experience = data for use in training neural networks
                       = data to learn from

    the discounted_return is the true return - ie Monte Carlo
    """

    def __init__(self, memory_length,
                 observation_space,
                 action_space,
                 reward_space,
                 discount_rate):

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
        self.episode_returns = []
        self.outputs = collections.defaultdict(list)
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
                # shouldn't need to do anything
                #  check value is already dummy
                assert (value == 0) or (value == 1)
            else:
                assert 1 == 0

            #  appending the scaled value onto the scaled array
            scaled_array = np.append(scaled_array, scaled).reshape(-1)

        return scaled_array

    def scale_reward(self, reward, space):
        """
        Helper function for make_machine_experience()
        Uses a space to scale the reward

        can probably move this into a parent
        """
        return self.normalize(reward, space.low, space.high)

    def make_machine_experience(self, exp, discounted_return=None):
        """

        Helper function for add_experience
        Scales a given experience tuple

        Discounted return is an optimal arg so that the scaled_exp array can
        be created at any time

        """
        scaled_obs = self.scale_array(exp[0], self.observation_space)
        scaled_reward = self.scale_reward(exp[2], self.reward_space)

        #  making an array for the scaled experience
        scaled_exp = np.array([scaled_obs,
                              exp[1],
                              scaled_reward,
                              None,
                              exp[4],
                              exp[5],
                              discounted_return])
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
        m_exp = self.make_machine_experience(exp)
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
        print(episode_experiences.shape)
        print('processing episode {} length {}'.format(episode_number,
                                                       len(episode_experiences)))

        #  using code from
        #  https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py#L65

        R = 0
        returns = []

        for exp in episode_experiences[::-1]:
            r = exp[2]
            R = r + self.discount_rate * R  # the Bellman equation
            returns.insert(0, R)

        #  now we normalize the episode returns
        returns = np.array(returns)
        avg_episode_return = returns.mean()
        returns = (returns - avg_episode_return) / (returns.std())

        #  now we have the normalized episode returns
        #  we can fill in the returns on our machine_experience
        new_exps = []
        for exp, rtn in itertools.zip_longest(episode_experiences, returns):
            exp[6] = rtn
            new_exps.append(exp)

        idx_array = np.arange(all_experiences.shape[0])
        assert idx_array.shape[0] == all_experiences.shape[0]
        episode_indicies = idx_array[episode_mask]
        start = episode_indicies[0]
        end = episode_indicies[-1] + 1
        self.machine_experiences[start:end] = new_exps

        self.episode_returns.append(avg_episode_return)

        print('return for episode {} was {}'.format(episode_number,
                                                    avg_episode_return))
        print('historic mean returns are {}'.format(np.mean(self.episode_returns)))
        print('historic median returns are {}'.format(np.median(self.episode_returns)))
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


# """
#
# code below is an old funciton i dont need anympre
#
#
# def process_episode(self, episode_number):
# """"""
# Calculates the discounted returns
#
# Inefficient as we loop over the entire episode list.
#
# TODO some sort of check that episode is actually over
# """"""
# #  gather the scaled experiences from the last episode
# #  we want to get access to the scaled reward
# episode_experiences, indicies, idx = [], [], 0
# for idx, exp in enumerate(self.machine_experiences, 0):
#     if exp[5] == episode_number:
#         episode_experiences.append(exp)
#         indicies.append(idx)
#
# #  we reverse our experience list so we can do an efficient backup
# episode_experiences.reverse()
#
# #  blank array to hold the returns
# rtns = np.zeros(len(episode_experiences))
# scaled_episode_experiences = []
#
# for j, exp in enumerate(episode_experiences):
#
#     if j == 0:
#         total_return = exp[2]
#
#     else:
#         total_return = exp[2] + self.discount_rate * rtns[j - 1]
#
#     #  note the experimental scaling!
#     scaled_return = total_return
#     rtns[j] = scaled_return
#
#     scaled_exp = np.array(exp[0],
#                           exp[1],
#                           exp[2],
#                           exp[3],
#                           exp[4],
#                           exp[5],
#                           total_return)
#
#     scaled_episode_experiences.append(scaled_exp)
#
# #  now we use our original indicies to reindex
# scaled_episode_experiences.reverse()
#
# for k, idx in enumerate(indicies):
#     self.machine_experiences[idx] = scaled_episode_experiences[k]
#
# assert len(self.experiences) == len(self.machine_experiences)
#
# return None
#
# """
#
# """
# code below works - but ignoring for now to allow meto focus on getting policy
# gradient learner working
#
#
#     def get_random_batch(self, batch_size):
#         """
#         """Gets a random batch of experiences."""
#         """
#         sample_size = min(batch_size, len(self.machine_experiences))
#
#         #  limiting the scaled_experiences list to the memory length
#         memory = self.experiences[-self.memory_length:]
#         scaled_memory = self.machine_experiences[-self.memory_length:]
#
#         assert len(memory) == len(scaled_memory
#                                   )
#         #  indicies for the batch
#         indicies = np.random.randint(low=0,
#                                      high=len(memory),
#                                      size=sample_size)
#
#         #  randomly sample from the memory & returns
#         memory_batch = [memory[i] for i in indicies]
#         scaled_memory_batch = [scaled_memory[i] for i in indicies]
#
#         observations = np.array(
#             [exp[0] for exp in scaled_memory_batch]).reshape(-1, len(self.observation_space))
#         actions = np.array([exp[1] for exp in memory_batch]
#                            ).reshape(-1, len(self.action_space))
#         returns = np.array(
#             [exp[6] for exp in scaled_memory_batch]).reshape(-1, 1)
#
#         assert observations.shape[0] == actions.shape[0]
#         assert observations.shape[0] == returns.shape[0]
#
#         return observations, actions, returns
# """
