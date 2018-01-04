import collections
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Memory(object):
    """
    An object to store and process experience.

    args
        observation_space (object) energy_py GlobalSpace
        action_space (object) energy_py GlobalSpace
        discount (float) the discount rate (aka gamma)
        memory_length (int) maximum number of experiences to store
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 discount,
                 memory_length):

        #  MDP info
        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = float(discount)
        self.memory_length = int(memory_length)

        self.reset()

    def reset(self):
        """
        Resets the memory internals.
        """
        #  keep count of the number of experiences
        self.num_exp = 0

        #  create an array for each type of info we want to store
        self.obs = np.array([], dtype=np.float32).reshape(-1,
                                                          self.observation_space.shape[0])

        self.actions = np.array([], dtype=np.float32).reshape(-1,
                                                              self.action_space.shape[0])

        self.rewards = np.array([], dtype=np.float32).reshape(-1, 1)

        self.next_obs = np.array([], dtype=np.float32).reshape(-1,
                                                               self.observation_space.shape[0])

        self.terminal = np.array([], dtype=bool).reshape(-1, 1)

        self.step = np.array([], dtype=np.int32).reshape(-1, 1)

        self.episode = np.array([], dtype=np.int32).reshape(-1, 1)

        #  an info dictionary to hold other info we might want to collect
        self.info = collections.defaultdict(list)

        #  an outputs dict to hold data we want to extract
        self.outputs = collections.defaultdict(list)

    def add_experience(self,
                       observation,
                       action,
                       reward,
                       next_observation,
                       terminal,
                       step,
                       episode):
        """
        Adds a single step of experience to the numpy arrays.

        args
            observation
            action
            reward
            next_observation
            terminal
            step
            episode
        """
        logger.debug('adding exp episode {} step {}'.format(episode, step))

        self.obs = np.append(self.obs, observation, axis=0)
        self.actions = np.append(self.actions, action, axis=0)

        reward = np.array(reward, dtype=np.float32).reshape(1, 1)
        self.rewards = np.append(self.rewards, reward, axis=0)

        self.next_obs = np.append(self.next_obs, next_observation, axis=0)

        terminal = np.array(terminal, dtype=np.bool).reshape(1, 1)
        self.terminal = np.append(self.terminal, terminal, axis=0)

        step = np.array(step, dtype=np.int32).reshape(1, 1)
        self.step = np.append(self.step, step, axis=0)

        episode = np.array(episode, dtype=np.int32).reshape(1, 1)
        self.episode = np.append(self.episode, episode, axis=0)

        self.num_exp += 1

    def calculate_returns(self, rewards):
        """
        Calculates the Monte Carlo discounted return

        args
            rewards (np.array) rewards we want to calculate the return for
        """
        R = 0  # return after state s
        returns = []  # return after next state s'

        #  reverse the list so that we can do a backup
        for r in rewards[::-1]:
            R = r + self.discount * R  # the Bellman equation
            returns.insert(0, R)

        #  turn into array, print out some statistics before we scale
        rtns = np.array(returns)
        logger.debug('total returns before scl {:.2f}'.format(rtns.sum()))
        logger.debug('mean returns before scl {:.2f}'.format(rtns.mean()))
        logger.debug('stdv returns before scl {:.2f}'.format(rtns.std()))

        return rtns.reshape(-1, 1)

    def get_episode_batch(self, episode_number):
        """
        Gets the experiences for a given episode

        args
            episode_number (int)

        returns
            batch_dict (dict)
                observations (np.array) shape=(samples, self.observation_dim)
                actions (np.array) shape=(samples, self.action_dim)
                rewards (np.array) shape=(samples, 1)
        """

        #  get the indicies of the episode we want
        episode_mask = np.where(self.episode == episode_number)[0]

        obs = self.obs[episode_mask]
        actions = self.actions[episode_mask]
        rewards = self.rewards[episode_mask]

        assert obs.shape[0] == actions.shape[0]
        assert obs.shape[0] == rewards.shape[0]

        assert not np.any(np.isnan(obs))
        assert not np.any(np.isnan(actions))
        assert not np.any(np.isnan(rewards))

        batch_dict = {'obs': obs,
                      'actions': actions,
                      'rewards': rewards}

        return batch_dict

    def get_random_batch(self, batch_size, save_batch=False):
        """
        Gets a random batch of experiences

        args
            batch_size (int)

        returns
            batch_dict (dict)
                obs (np.array) shape=(samples, self.observation_dim)
                actions (np.array) shape=(samples, self.action_dim)
                rewards (np.array) shape=(samples, 1)
                next_obs (np.array) shape=(samples, self.observation_dim)
                terminal (np.array) shape=(samples, 1)
        """
        sample_size = min(batch_size, self.num_exp)
        logger.debug('getting batch size {} from memory'.format(sample_size))

        #  indicies for the batch
        lower_bound = max(0, self.num_exp - self.memory_length)
        indicies = np.random.randint(low=lower_bound,
                                     high=self.num_exp-1,
                                     size=sample_size)

        #  sample from the memory using these indicies
        obs = self.obs[indicies]
        actions = self.actions[indicies]
        rewards = self.rewards[indicies]
        next_obs = self.next_obs[indicies]
        terminal = self.terminal[indicies]

        batch_dict = {'obs': obs,
                      'actions': actions,
                      'rewards': rewards,
                      'next_obs': next_obs,
                      'terminal': terminal}

        return batch_dict

    def output_results(self):
        """
        Extract data from the memory

        returns
            self.outputs (dict) includes self.info
        """
        self.outputs['info'] = self.info

        obs = pd.DataFrame(self.obs,
                           columns=['obs_{}'.format(i) for i in
                                    range(self.obs.shape[1])])

        act = pd.DataFrame(self.actions,
                           columns=['act_{}'.format(i) for i in
                                    range(self.actions.shape[1])])

        rew = pd.DataFrame(self.rewards,
                           columns=['reward'])

        next_obs = pd.DataFrame(self.next_obs,
                                columns=['next_obs_{}'.format(i) for i in
                                         range(self.next_obs.shape[1])])

        step = pd.DataFrame(self.step,
                            columns=['step'])

        episode = pd.DataFrame(self.episode,
                               columns=['episode'])

        #  make a dataframe on a step by step basis
        df_stp = pd.concat([obs,
                            act,
                            rew,
                            next_obs,
                            step,
                            episode], axis=1)

        df_stp.set_index('episode', drop=True, inplace=True)

        #  make a dataframe on an episodic basis
        df_ep = df_stp.groupby(by=['episode'], axis=0).sum()
        reward = df_ep.loc[:, 'reward']
        #  add statistics into the episodic dataframe
        df_ep.loc[:, 'cum max reward'] = reward.cummax()
        #  set the window at 10% of the data
        window = max(int(df_ep.shape[0]*0.1), 2)
        df_ep.loc[:, 'rolling mean'] = reward.rolling(window,
                                                      min_periods=2).mean()

        df_ep.loc[:, 'rolling std'] = reward.rolling(window,
                                                     min_periods=2).std()

        #  saving data in the output_dict
        self.outputs['df_stp'] = df_stp
        self.outputs['df_ep'] = df_ep

        return self.outputs
