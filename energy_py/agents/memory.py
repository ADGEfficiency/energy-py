from collections import defaultdict, deque, namedtuple
from random import sample

import numpy as np

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
    """
    R = 0  # return after state s
    returns = []  # return after next state s'

    #  reverse the list so that we can do a backup
    for r in rewards[::-1]:
        R = r + discount * R  # the Bellman equation
        returns.insert(0, R)

    #  turn into array
    return np.array(returns).reshape(-1, 1)


class Memory(object):
    """
    Base class for agent memories
    """

    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 size):

        self.size = int(size)

        #  use a dict to hold the shapes
        #  we can use this to eaisly reshape batches of experience
        self.shapes = {'observations': observation_space_shape,
                       'actions': action_space_shape,
                       'rewards': (1,),
                       'next_observations': observation_space_shape,
                       'terminal': (1,)}


class DequeMemory(Memory):
    """
    Implementation of an experience replay memory based on a deque.

    A single sample of experience is held in a namedtuple.
    Sequences of experience are kept in a deque.
    Batches are randomly sampled from this deque.
    """

    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 size):

        super(DequeMemory, self).__init__(observation_space_shape,
                                          action_space_shape,
                                          size)

        self.experiences = deque(maxlen=self.size)

    def __repr__(self): return '<class Memory len={}>'.format(len(self))

    def __len__(self): return len(self.experiences)

    def reset(self): raise NotImplementedError()

    def remember(self, observation, action, reward,
                 next_observation, terminal):
        """
        Adds experience to the memory.

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

        for key, data in batch_dict.items():
            batch_dict[key] = np.array(data).reshape(-1, *self.shapes[key])

        return batch_dict


class ArrayMemory(object):
    """
    Implementation of an experience memory replay based on numpy arrays.

    A memory based on individual numpy arrays for each dimension of the
    (s, a, r, s') experience tuple.
    """

    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 size):

        self.size = size

        #  use a dict to hold the shapes
        #  we can use this to eaisly reshape batches of experience
        self.shapes = {'observations': observation_space_shape,
                       'actions': action_space_shape,
                       'rewards': (1,),
                       'next_observations': observation_space_shape,
                       'terminal': (1,)}

        self.reset()

    def __repr__(self): return '<class Memory len={}>'.format(len(self))

    def __len__(self): return self.count

    def reset(self):

        self.count = 0

        self.obs = np.empty((self.size, *self.shapes['observations']))
        self.acts = np.empty((self.size, *self.shapes['actions']))
        self.rews = np.empty((self.size, *self.shapes['rewards']))
        self.n_obs = np.empty((self.size, *self.shapes['next_observations']))
        self.term = np.empty((self.size, *self.shapes['terminal']), dtype=bool)

    def remember(self, observation, action, reward,
                 next_observation, terminal):
        """
        Adds experience to the memory.

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
