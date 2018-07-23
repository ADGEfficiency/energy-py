from collections import namedtuple
import logging

import numpy as np

from energy_py.common.utils import dump_pickle

logger = logging.getLogger(__name__)


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


class BaseMemory(object):
    """
    Base class for agent memories

    The shapes dictionary is used to reshape experience dimensions
    """
    def __init__(
            self,
            env,
            size,
    ):

        self.size = int(size)
        self.shapes = {
            'observation': env.observation_space.shape,
            'action': env.action_space.shape,
            'reward': (1,),
            'next_observation': env.observation_space.shape,
            'done': (1,),
            'importance_weight': (1,),
            'indexes': (1,)  #  does this do anything ? TODO
        }

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
             'reward': np.array(batch_size),
             'next_observation': np.array(batch_size, *obs_shape),
             'done': np.array(batch_size)}
        """
        batch_dict = {}

        for field in Experience._fields:
            arr = np.array([getattr(e, field) for e in batch])
            batch_dict[field] = arr.reshape(len(batch), *self.shapes[field])

        return batch_dict

    def save(self, path):
        """ saves the memory to a pickle """
        logger.info('Saving memory to {}'.format(path))
        dump_pickle(self, path)
