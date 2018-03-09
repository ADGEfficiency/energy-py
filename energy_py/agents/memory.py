from collections import defaultdict, deque, namedtuple
from random import sample

import numpy as np

#  use a namedtuple to store a single sample of experience
Experience = namedtuple('Experience', ['observation',
                                       'action',
                                       'reward',
                                       'next_observation',
                                       'terminal'])


def calculate_returns(rewards, discount):
    """
    Calculates the Monte Carlo discounted return.

    args
        rewards (np.array) rewards we want to calculate the return for

    returns
        returns (np.array) the return for each state
    """
    R = 0  # return after state s
    returns = []  # return after next state s'

    #  reverse the list so that we can do a backup
    for r in rewards[::-1]:
        R = r + discount * R  # the Bellman equation
        returns.insert(0, R)

    return np.array(returns).reshape(-1, 1)


class Memory(object):
    """
    Base class for agent memories

    Sets the agent size and creates a shapes dictionary.  The shapes
    dictionary can be used to reshape arrays stored in the memory

    args
        size (int)
        obs_shape (tuple)
        action_shape (tuple)
    """
    def __init__(self,
                 size,
                 obs_shape,
                 action_shape):

        self.size = int(size)

        #  use a dict to hold the shapes
        #  we can use this to eaisly reshape batches of experience
        self.shapes = {'observations': obs_shape,
                       'actions': action_shape,
                       'rewards': (1,),
                       'next_observations': obs_shape,
                       'terminal': (1,),
                       'importance_weights': (1,),
                       'indexes': (1,)}


class DequeMemory(Memory):
    """
    Implementation of an experience replay memory based on a deque.

    A single sample of experience is held in a namedtuple.
    Sequences of experience are kept in a deque.
    Batches are randomly sampled from this deque.
    """

    def __init__(self,
                 size,
                 obs_shape,
                 action_shape):

        super(DequeMemory, self).__init__(size,
                                          obs_shape,
                                          action_shape)

        self.experiences = deque(maxlen=self.size)

    def __repr__(self):
        return '<class DequeMemory len={}>'.format(len(self))

    def __len__(self):
        return len(self.experiences)

    def remember(self, observation, action, reward,
                 next_observation, terminal):
        """
        Adds experience to the memory

        args
            observation
            action
            reward
            next_observation
            terminal
        """
        #  create an experience named tuple
        #  add the experience to our deque
        #  the deque automatically keeps our memory at the correct size
        self.experiences.append(Experience(observation,
                                           action,
                                           reward,
                                           next_observation,
                                           terminal))

    def get_batch(self, batch_size):
        """
        Samples a batch randomly from the memory.

        args
            batch_size (int)

        returns
            batch_dict (dict)
        """
        sample_size = min(batch_size, len(self))
        batch = sample(self.experiences, sample_size)
        batch_dict = defaultdict(list)

        for exp in batch:
            batch_dict['observations'].append(exp.observation)
            batch_dict['actions'].append(exp.action)
            batch_dict['rewards'].append(exp.reward)
            batch_dict['next_observations'].append(exp.next_observation)
            batch_dict['terminal'].append(exp.terminal)

        #  use the shapes dictionary to reshape our arrays
        for key, data in batch_dict.items():
            batch_dict[key] = np.array(data).reshape(-1, *self.shapes[key])

        return batch_dict


class ArrayMemory(Memory):
    """
    Implementation of an experience memory replay based on numpy arrays.

    A memory based on individual numpy arrays for each dimension of the
    (s, a, r, s') experience tuple.
    """

    def __init__(self,
                 size,
                 obs_shape,
                 action_shape):

        super(ArrayMemory, self).__init__(size,
                                          obs_shape,
                                          action_shape)

        #  create one np array for each dimension of experience
        #  the first dimension of these arrays is the memory dimension
        self.obs = np.empty((self.size, *self.shapes['observations']))
        self.acts = np.empty((self.size, *self.shapes['actions']))
        self.rews = np.empty((self.size, *self.shapes['rewards']))
        self.n_obs = np.empty((self.size, *self.shapes['next_observations']))
        self.term = np.empty((self.size, *self.shapes['terminal']), dtype=bool)

        #  keep a counter to index the numpy arrays
        self.count = 0

    def __repr__(self):
        return '<class ArrayMemory len={}>'.format(len(self))

    def __len__(self):
        return self.count

    def remember(self, observation, action, reward,
                 next_observation, terminal):
        """
        Adds experience to the memory

        args
            observation
            action
            reward
            next_observation
            terminal
        """
        self.obs[self.count] = observation
        self.acts[self.count] = action
        self.rews[self.count] = reward
        self.n_obs[self.count] = next_observation
        self.term[self.count] = terminal

        #  conditional to reset the counter once we end of the array
        if self.count == self.size:
            self.count = 0
        else:
            self.count += 1

    def get_batch(self, batch_size):
        """
        Samples a batch randomly from the memory.

        args
            batch_size (int)

        returns
            batch_dict (dict)
        """
        sample_size = min(batch_size, len(self))
        indicies = np.random.randint(len(self), size=sample_size)

        batch_dict = {'observations': self.obs[indicies],
                      'actions': self.acts[indicies],
                      'rewards': self.rews[indicies],
                      'next_observations': self.n_obs[indicies],
                      'terminal': self.term[indicies]}

        return batch_dict
