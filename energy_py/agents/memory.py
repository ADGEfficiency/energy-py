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
    A class to hold the memory of an agent.
    """
    def __init__(self, memory_length,
                       observation_space,
                       action_space,
                       reward_space,
                       discount_factor):

        super().__init__()
        self.memory_length     = memory_length
        self.observation_space = observation_space
        self.action_space      = action_space
        self.reward_space      = reward_space
        self.discount_factor   = discount_factor

        #  a named tuple to hold experience
        self.Experience = collections.namedtuple('experience', 'observation, action, reward, next_observation, step, episode')
        self.Scaled_Experience = collections.namedtuple('experience', 'observation, action, reward, next_observation, step, episode, discounted_return')

        self.reset()

    def reset(self):
        self.experiences        = []
        self.scaled_experiences = []
        self.discounted_returns = np.array([])
        self.outputs  = collections.defaultdict(list)
        self.losses = []

    def normalize(self, value, low, high):
        """
        Helper function
        Normalizes a value
        """
        #  if statement to catch the constant value case
        if low == high:
            normalized = 0
        else:
            max_range = high - low
            normalized = (value - low) / max_range
        return np.array(normalized)

    def scale_array(self, array, space, scaler_fctn):
        """
        Helper function
        Uses the space & a given function to scale an array
        Default scaler is to normalize

        Used to scale the observation and action
        """
        #  empty numpy array
        scaled_array = np.array([])
        #  iterate across the array values & corresponding space object
        for value, spc in itertools.zip_longest(array, space):
            #  use the fctn to transform the array
            scaled = scaler_fctn(value,
                                 spc.low,
                                 spc.high)
            #  appending the scaled value onto the scaled array
            scaled_array = np.append(scaled_array, scaled)
        assert array.shape == scaled_array.shape
        return scaled_array

    def scale_reward(self, reward, space, scaler_fctn):
        """
        Helper function
        Uses a space to scale the reward
        """
        return scaler_fctn(reward, space.low, space.high)

    def scale_experience(self, exp, discounted_return=None):
        #  scaling the observation
        #  could can use the scaled next_obs of the previous experience
        #  to set the obs entry of the new experience (more efficient)

        #  here I do it a simple way - scaling the observation
        scaled_obs = self.scale_array(exp.observation, self.observation_space, self.normalize)
        #  scaling the action
        scaled_action = self.scale_array(exp.action, self.action_space, self.normalize)
        #  scaling the reward
        scaled_reward = self.scale_reward(exp.reward, self.reward_space, self.normalize)
        #  scaling the next_observation
        #  if statement deals with the lost bit of experience
        if exp.next_observation is False:
            scaled_next_observation = exp.next_observation
        else:
            scaled_next_observation = self.scale_array(exp.next_observation, self.observation_space, self.normalize)

        #  making a named tuple for the scaled experience
        scaled_exp = self.Scaled_Experience(scaled_obs,
                                            scaled_action,
                                            scaled_reward,
                                            scaled_next_observation,
                                            exp.step,
                                            exp.episode,
                                            discounted_return)
        return scaled_exp

    def add_experience(self, observation, action, reward, next_observation, step, episode):
        """
        Adds a single step of experience to the experiences list.
        """
        exp = self.Experience(observation, action, reward, next_observation, step, episode)
        self.experiences.append(exp)

        scaled_exp = self.scale_experience(exp)
        self.scaled_experiences.append(scaled_exp)
        return None

    def process_episode(self, episode_number):
        """
        Calculates the discounted returns

        Should only be done once a episode is finished - TODO a check
        """
        episode_experiences = [exp for exp in self.scaled_experiences if exp.episode == episode_number]
        scaled_episode_experiences = []

        for i, experience in enumerate(episode_experiences):
            discounted_return = sum(self.discount_factor**j * exp.reward for j, exp in enumerate(episode_experiences[i:]))
            scaled_exp = self.scale_experience(experience, discounted_return)
            scaled_episode_experiences.append(scaled_exp)

            idx = -(len(episode_experiences) - i)
            self.scaled_experiences[idx] = scaled_exp

        assert len(episode_experiences) == len(scaled_episode_experiences)
        return None

    def get_batch(self, batch_size):
        """
        Gets a random batch of experiences.
        """

        sample_size = min(batch_size, len(self.scaled_experiences))
        #  limiting the scaled_experiences list to the memory length
        memory = self.experiences[-self.memory_length:]
        scaled_memory = self.scaled_experiences[-self.memory_length:]

        assert len(memory) == len(scaled_memory
                                  )
        #  indicies for the batch
        indicies = np.random.randint(low=0,
                                     high=len(memory),
                                     size=sample_size)
        #  randomly sample from the memory & returns
        memory_batch = [memory[i] for i in indicies]
        scaled_memory_batch = [scaled_memory[i] for i in indicies]

        #  note that we dont take the scaled returns!!!
        observations = np.array([exp.observation for exp in scaled_memory_batch]).reshape(-1, len(self.observation_space))
        actions = np.array([exp.action for exp in memory_batch]).reshape(-1, len(self.action_space))
        returns = np.array([exp.discounted_return for exp in scaled_memory_batch]).reshape(-1, 1)

        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == returns.shape[0]

        return observations, actions, returns

    def make_dataframes(self):
        """
        Helper function for self.output_results()

        Creates two dataframes
        'dataframe_steps'    = dataframe on a step frequency
        'dataframe_episodic' = dataframe on a episodic frequency
        """
        #  create lists on a step by step basis
        rewards = [exp.reward for exp in self.experiences]
        episodes = [exp.episode for exp in self.experiences]
        actions = [exp.action[0] for exp in self.experiences]

        df_dict = {'reward':rewards,
                   'episode':episodes,
                   'action':actions}

        dataframe_steps = pd.DataFrame.from_dict(df_dict)

        dataframe_episodic = dataframe_steps.groupby(by=['episode'],
                                                  axis=0).sum()

        if self.losses:
            dataframe_episodic.loc[:, 'loss'] = self.losses

        dataframe_steps.set_index('episode', drop=True, inplace=True)
        # dataframe_episodic.set_index('episode', drop=True, inplace=True)

        path_steps = os.path.join(self.base_path, 'agent_df_steps.csv')
        ensure_dir(path_steps)
        dataframe_steps.to_csv(path_steps)

        path_episodic = os.path.join(self.base_path, 'agent_df_episodic.csv')
        ensure_dir(path_episodic)
        dataframe_episodic.to_csv(path_episodic)
        return dataframe_steps,  dataframe_episodic

    def make_returns_fig(self):
        """
        Makes a plot of undiscounted reward per episode
        """
        fig = self.make_time_series_fig(self.outputs['dataframe_episodic'],
                                                          cols=['reward'],
                                                          ylabel='Undiscounted total reward per episode',
                                                          xlabel='Episode')

        path = os.path.join(self.base_path, 'return_per_episode.png')
        ensure_dir(path)
        fig.savefig(path)
        return fig

    def make_loss_fig(self):
        """
        Makes a plot of undiscounted reward per episode
        """
        fig = self.make_time_series_fig(self.outputs['dataframe_episodic'],
                                                          cols=['loss'],
                                                          ylabel='Loss',
                                                          xlabel='Episode',
                                                          ylim=[-100, 0])

        path = os.path.join(self.base_path, 'loss_per_episode.png')
        ensure_dir(path)
        fig.savefig(path)
        return fig

    def _output_results(self):
        """
        The main function to output results from the memory

        Overridden method of the Visualizer parent class

        Envionsed to run after all episodes are finished
        """
        #  making the two summary dataframes
        self.outputs['dataframe_steps'], self.outputs['dataframe_episodic'] = self.make_dataframes()

        self.outputs['returns_fig'] = self.make_returns_fig()

        if self.losses:
            self.outputs['loss_fig'] = self.make_loss_fig()
        return self.outputs
