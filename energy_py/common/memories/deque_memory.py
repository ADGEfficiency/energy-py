from collections import deque
import random

from energy_py.common.memories.memory import BaseMemory, Experience


class DequeMemory(BaseMemory):
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


