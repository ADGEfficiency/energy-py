import collections
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


#  use a named_tuple to store a single step of experience
Experience = collections.namedtuple('experience', ['state',
                                                   'action',
                                                   'reward',
                                                   'next_state',
                                                   'terminal',
                                                   'step',
                                                   'episode'])
class Memory(object):
    """
    An object to store and process experience.

    A deque is used store experience tuples.
    The experience tuples are named tuples.
    The deque is dumped to disk in text files everytime it is wiped over.

    I experimented with antother memory structure based on using one
    numpy array for state, one for action etc.  The deque structure
    was much faster.  You can see this work in energy_py/notebooks

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

        self.shapes = {'obs': self.observation_space.shape,
                       'actions': self.action_space.shape,
                       'rewards': (1,),
                       'next_observation': self.observation_space.shape,
                       'terminal': (1,)}

        self.reset()

    def reset(self):
        """
        Resets the memory internals.
        """
        #  keep count of the number of experiences
        self.count = 0

        #  use a deque to store experience
        self.experiences = collections.deque(maxlen=self.memory_length)

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

        self.experiences.append(Experience(observation,
                                           action,
                                           reward,
                                           next_observation,
                                           terminal,
                                           step,
                                           episode))

        self.count += 1
        #  check if we need to save the memory to disk
        if self.count % self.memory_length == 0:
            self.dump_memory()

    def dump_memory(self):
        """

        """
        pass

    @staticmethod
    def batch_to_dict(batch):

        batch_dict = collections.defaultdict(list)
        for exp in batch:
            batch_dict['obs'].append(exp.observation)
            batch_dict['actions'].append(exp.action)
            batch_dict['rewards'].append(exp.reward)
            batch_dict['next_obs'].append(exp.next_observation)
            batch_dict['terminal'].append(exp.terminal)

        for key, data in batch_dict.items():
            data = np.array(data).reshape(sample_size, *self.shapes[key])
            batch_dict[key] = data        
            assert not np.any(np.isnan(data))

        return batch_dict


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
        eps = np.array([exp.episode for exp in self.experiences]).reshape(-1,1)
        episode_mask = np.where(eps == episode_number)[0]

        episode_experiences = self.experiences[episode_mask]

        return self.batch_to_dict(episode_experiences)

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

        batch = random.sample(self.experiences, sample_size)

        return self.batch_to_dict(batch)

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
