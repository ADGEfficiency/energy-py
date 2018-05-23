"""
Basic memory structures to hold an agent's experiences

The memory remember() method is called from the agent - this means
that the agent can preprocess dimensions of experience (i.e. reward clip)
before the experience is remembered

Experience namedtuple is used to hold a single sample of experience

calculate_returns()
- function to calculate the Monte Carlo discounted return

class Memory
- the base class for memories

class DequeMemory
- is the fastest impelmentation
- uses a deque to store experience as namedtuples (one per step)
- sampling by indexing experience and unpacking into arrays

class ArrayMemory
- stores each dimension of experience (state, action etc)
  in separate numpy arrays
- sampling experience is done by indexing each array
"""

from collections import defaultdict, deque, namedtuple
import random

import numpy as np


#  use a namedtuple to store a single sample of experience
Experience = namedtuple('Experience', ['observation',
                                       'action',
                                       'reward',
                                       'next_observation',
                                       'done'])


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

    args
        size (int)
        obs_shape (tuple)
        action_shape (tuple)

    Shapes dictionary used to reshape experience dimensions
    """
    def __init__(self,
                 size,
                 obs_shape,
                 action_shape):

        self.size = int(size)

        self.shapes = {'observation': obs_shape,
                       'action': action_shape,
                       'reward': (1,),
                       'next_observation': obs_shape,
                       'done': (1,),
                       'importance_weight': (1,),
                       'indexes': (1,)}

    def make_batch_dict(self, batch):
        """
        Takes a list of experiences and converts into a dictionary

        args
            batch (list)

        returns
            batch_dict (dict)

        Batch converted into batch_dict:
            {'observation': np.array(batch_size, *obs_shape,
             'action': np.array(batch_size, *act_shape),
             'reward': np.array(batch_size, 1),
             'next_observation': np.array(batch_size, *obs_shape),
             'done': np.array(batch_size, 1)}
        """
        batch_dict = {} 

        for field in Experience._fields:
            arr = np.array([getattr(e, field) for e in batch])
            batch_dict[field] = arr.reshape(len(batch), *self.shapes[field])
        return batch_dict

class DequeMemory(Memory):
    """
    Experience replay memory based on a deque

    args
        size (int)
        obs_shape (tuple)
        action_shape (tuple)

    A single sample of experience is held in a namedtuple
    Sequences of experience are kept in a deque
    Batches are randomly sampled from this deque

    This requires unpacking the deques for every batch
    - small batch sizes mean this isn't horrifically expensive
    """

    def __init__(self,
                 size,
                 obs_shape,
                 action_shape):

        super().__init__(size,
                         obs_shape,
                         action_shape)

        self.experiences = deque(maxlen=self.size)

    def __repr__(self):
        return '<class DequeMemory size={}>'.format(self.size)

    def __len__(self):
        return len(self.experiences)

    def remember(self, observation, action, reward, next_observation, done):
        """
        Adds experience to the memory

        args
            observation
            action
            reward
            next_observation
            done

        Deque automatically keeps memory at correct size
        """
        self.experiences.append(Experience(observation,
                                           action,
                                           reward,
                                           next_observation,
                                           done))

    def get_batch(self, batch_size):
        """
        Samples a batch randomly from the memory

        args
            batch_size (int)

        returns
            batch_dict (dict)
        """
        sample_size = min(batch_size, len(self))
        batch = random.sample(self.experiences, sample_size)

        return self.make_batch_dict(batch)


class ArrayMemory(Memory):
    """
    Experience memory replay based on numpy arrays

    args
        size (int)
        obs_shape (tuple)
        action_shape (tuple)

    Individual numpy arrays for each dimension of experience

    First dimension of each array is the memory dimension
    """

    def __init__(self,
                 size,
                 obs_shape,
                 action_shape):

        super().__init__(size,
                         obs_shape,
                         action_shape)

        self.obs = np.empty((self.size, *self.shapes['observation']))
        self.acts = np.empty((self.size, *self.shapes['action']))
        self.rews = np.empty((self.size, *self.shapes['reward']))
        self.n_obs = np.empty((self.size, *self.shapes['next_observation']))
        self.term = np.empty((self.size, *self.shapes['done']), dtype=bool)

        self.count = 0

    def __repr__(self):
        return '<class ArrayMemory size={}>'.format(self.size)

    def __len__(self):
        return self.count

    def remember(self, observation, action, reward, next_observation, done):
        """
        Adds experience to the memory

        args
            observation
            action
            reward
            next_observation
            done
        """
        self.obs[self.count] = observation
        self.acts[self.count] = action
        self.rews[self.count] = reward
        self.n_obs[self.count] = next_observation
        self.term[self.count] = done

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
                      'done': self.term[indicies]}

        return batch_dict
