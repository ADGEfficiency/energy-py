import collections
import itertools
import os

import numpy as np
import pandas as pd

from energy_py.main.scripts.utils import Utils


class Agent_Memory(Utils):
    """
    Purpose of this class is to
        store the experiences of the agent
        process experiences for use by the agent to act or learn from

    The memory of the agent is two lists of experience numpy arrays
        self.experiences = data as observed
        self.machine_experiences = data for use by neural networks

    The two experience numpy arrays hold the following data
        experience = (observation,          0
                      action,               1
                      reward,               2
                      next_observation,     3
                      step,                 4
                      episode)              5

        machine_experience = (observation,       0
                              action,            1
                              reward,            2
                              next_observation,  3
                              step,              4
                              episode,           5
                              discounted_return) 6

      the discounted_return is the Monte Carlo return
    """

    def __init__(self, memory_length,
                       observation_space,
                       action_space,
                       reward_space,
                       discount,
                       verbose=False):

        super().__init__(verbose)
        self.memory_length = memory_length
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.discount = discount
        self.verbose = verbose

        self.reset()

    def reset(self):
        """
        Resets the memory object
        """
        self.experiences = []
        self.machine_experiences = []

        #  TODO a cleaner way to keep track of these statistics
        self.losses = [0] #  in case we print results before we train
        self.epsilons = []

    def make_machine_experience(self, exp, normalize_reward):
        """
        Helper function for add_experience
        Scales a given experience tuple

        Discounted return not updated here as we don't know it yet!
        i.e. this function is used within episode
        """
        scaled_obs = self.scale_array(exp[0],
                                      self.observation_space)

        scaled_action = self.scale_array(exp[1],
                                      self.action_space)

        if normalize_reward:
            reward = self.normalize(exp[2],
                                    self.reward_space.low,
                                    self.reward_space.high)
            reward = reward.reshape(1, 1)

        else:
            reward = exp[2]

        #  this if statement is needed because for the terminal state
        #  the next observation = False
        if exp[3].all() == -999999:
            scaled_next_obs = exp[3]
        else:
            scaled_next_obs = self.scale_array(exp[3],
                                               self.observation_space)

        #  making an array for the scaled experience
        scaled_exp = np.array([scaled_obs,
                               scaled_action,
                               reward,
                               scaled_next_obs,
                               exp[4],  # step
                               exp[5],  # episode number
                               None])   # the Monte Carlo return
        return scaled_exp

    def add_experience(self, observation, action, reward, next_observation,
                       step, episode, normalize_reward=True):
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
        m_exp = self.make_machine_experience(exp, normalize_reward)

        #  add experiences to the memory
        self.experiences.append(exp)
        self.machine_experiences.append(m_exp)
        return None

    def calc_returns(self, episode_number, normalize_return):
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
            R = r + self.discount * R  # the Bellman equation
            returns.insert(0, R)
            rewards.append(r)

        #  now we normalize the episode returns
        rtns = np.array(returns)

        self.verbose_print('episode {}'.format(episode_number),
                           'total returns before scaling {:.2f}'.format(rtns.sum()),
                           'mean returns before scaling {:.2f}'.format(rtns.mean()),
                           'stdv returns before scaling {:.2f}'.format(rtns.std()))

        if normalize_return:
            rtns = rtns / rtns.std()
            #rtns = (rtns - rtns.mean()) / (rtns.std())
            #rtns = (rtns - rtns.min()) / (rtns.max() - rtns.min())

        #  now we have the episode returns
        #  we can fill in the returns each experience in machine_experience
        #  for this episode
        new_exps = []
        assert len(episode_experiences) == len(rtns)
        for exp, rtn in zip(episode_experiences, rtns):
            exp[6] = rtn
            new_exps.append(exp)

        idx_array = np.arange(all_experiences.shape[0])
        assert idx_array.shape[0] == all_experiences.shape[0]
        episode_indicies = idx_array[episode_mask]
        start = episode_indicies[0]
        end = episode_indicies[-1] + 1
        self.machine_experiences[start:end] = new_exps

        return None

    def get_episode_batch(self, episode_number, scaled_actions):
        """
        Gets the experiences for a given episode.
        """

        all_experiences = np.array(self.machine_experiences)
        assert all_experiences.shape[0] == len(self.machine_experiences)

        #  use boolean indexing to get experiences from last episode
        episode_mask = [all_experiences[:, 5] == episode_number]
        episode_experiences = all_experiences[episode_mask]

        observations = np.array(
            [exp[0] for exp in episode_experiences]).reshape(-1, len(self.observation_space))

        returns = np.array(
            [exp[6] for exp in episode_experiences]).reshape(-1, 1)
        
        #  deal with the case of zero reward for all sampls
        if np.any(np.isnan(returns)):
            returns = np.zeros(shape=returns.shape)

        #  if we require unscaled actions then we need to use the experiences list
        #  this will be needed for policy gradients where we need the logprob of the
        #  action taken
        if scaled_actions:
            actions = np.array([exp[1] for exp in episode_experiences]).reshape(-1, len(self.action_space))
        else:
            unscaled_exp = np.array(self.experiences)[episode_mask]
            actions = np.array([exp[1] for exp in unscaled_exp]).reshape(-1, len(self.action_space))

        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == returns.shape[0]

        assert not np.any(np.isnan(observations))
        assert not np.any(np.isnan(actions))
        assert not np.any(np.isnan(returns))

        return observations, actions, returns

    def get_random_batch(self, batch_size, save_batch=False):
        """
        Gets a random batch of experiences
        Uses machine_experiences

        """
        sample_size = min(batch_size, len(self.machine_experiences))

        #  limiting to the memory length
        mach_memory = self.machine_experiences[-self.memory_length:]

        #  indicies for the batch
        indicies = np.random.randint(low=0,
                                     high=len(mach_memory),
                                     size=sample_size)

        #  randomly sample from the memory & returns
        mach_exp_batch = [mach_memory[i] for i in indicies]

        obs = [exp[0] for exp in mach_exp_batch]
        acts = [exp[1] for exp in mach_exp_batch]
        rwrds = [exp[2] for exp in mach_exp_batch]
        next_obs = [exp[3] for exp in mach_exp_batch]

        observations = np.array(obs).reshape(sample_size,
                                             len(self.observation_space))

        actions = np.array(acts).reshape(sample_size, len(self.action_space))

        rewards = np.array(rwrds).reshape(sample_size, 1)


        next_observations = np.array(next_obs).reshape(sample_size,
                                                       len(self.observation_space))
        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == rewards.shape[0]
        assert observations.shape[0] == next_observations.shape[0]

        assert not np.any(np.isnan(observations))
        assert not np.any(np.isnan(actions))
        assert not np.any(np.isnan(rewards))
        assert not np.any(np.isnan(next_observations))

        if save_batch:
            self.verbose_print('saving training batch to disk')
            #  TODO
        return observations, actions, rewards, next_observations

    def output_results(self):
        """
        Creates two dataframes
        'dataframe_steps'    = dataframe on a step frequency
        'dataframe_episodic' = dataframe on a episodic frequency
        """
        #  create lists on a step by step basis
        print('agent memory is making dataframes')
        assert len(self.experiences) == len(self.machine_experiences)

        ep, stp, obs, act, rew, nxt_obs = [], [], [], [], [], []
        mach_obs, mach_act, mach_rew, mach_nxt_obs, dis_ret = [], [], [], [], []
        for exp, mach_exp in itertools.zip_longest(self.experiences, self.machine_experiences):
            obs.append(exp[0])
            act.append(exp[1])
            rew.append(exp[2])
            nxt_obs.append(exp[3])
            stp.append(exp[4])
            ep.append(exp[5])

            mach_obs.append(mach_exp[0])
            mach_act.append(mach_exp[1])
            mach_rew.append(mach_exp[2])
            mach_nxt_obs.append(mach_exp[3])
            dis_ret.append(mach_exp[6])

        df_dict = {
                   'episode':ep,
                   'step':stp,
                   'observation':obs,
                   'action':act,
                   'reward':rew,
                   'next_observation':nxt_obs,
                   'scaled_reward':mach_rew,
                   'discounted_return':dis_ret,
                   'scaled_observation':mach_obs,
                   'scaled_action':mach_act,
                   'scaled_reward':mach_rew,
                   'scaled_next_observation':mach_nxt_obs,
                   }

        dataframe_steps = pd.DataFrame.from_dict(df_dict)

        dataframe_episodic = dataframe_steps.groupby(by=['episode'],
                                                  axis=0).sum()

        max_reward = dataframe_episodic.loc[:, 'reward'].cummax()
        dataframe_episodic.loc[:, 'cum_max_reward'] = max_reward

        window = max(int(dataframe_episodic.shape[0] * 0.1), 2)
        rolling = dataframe_episodic.loc[:, 'reward'].rolling(window=window,
                                                              min_periods=1,
                                                              center=False)
        dataframe_episodic.loc[:, 'rolling_mean'] = rolling.mean()

        # dataframe_episodic.loc[:, 'epsilon'] = self.epsilons

        training_history = pd.DataFrame(self.losses,  columns=['loss'])

        dataframe_steps.set_index('episode', drop=True, inplace=True)

        output_dict = {'dataframe_steps' : dataframe_steps,
                       'dataframe_episodic' : dataframe_episodic,
                       'training_history'   : training_history}

        return output_dict
