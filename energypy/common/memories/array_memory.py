import numpy as np

from energypy.common.memories.memory import BaseMemory


class ArrayMemory(BaseMemory):
    """
    Experience replay memory where experience is randomly sampled

    Individual numpy arrays for each dimension of experience

    First dimension is the number of samples of experience
    """
    def __init__(
            self,
            env,
            size=10000
    ):
        super().__init__(env, size)
        self.type = 'array'

        empty = np.empty
        self.obs = empty((self.size, *self.shapes['observation']))
        self.acts = empty((self.size, *self.shapes['action']))
        self.rews = empty((self.size, *self.shapes['reward']))
        self.n_obs = empty((self.size, *self.shapes['next_observation']))
        self.term = empty((self.size, *self.shapes['done']), dtype=bool)

        self.cursor = 0
        self.count = 0

    def __repr__(self):
        return '<class ArrayMemory size={}>'.format(self.size)

    def __len__(self):
        return min(self.count, self.size)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.acts[idx].reshape(1, *self.shapes['action']),
            self.rews[idx],
            self.n_obs[idx],
            self.term[idx]
        )

    def remember(self, observation, action, reward, next_observation, done):
        """ adds experience to the memory """
        ar = np.array

        self.obs[self.cursor] = ar(observation).reshape(
            1, *self.shapes['observation'])

        self.acts[self.cursor] = ar(action).reshape(
            1, *self.shapes['action'])

        self.rews[self.cursor] = ar(reward).reshape(
            1, *self.shapes['reward'])

        self.n_obs[self.cursor] = ar(next_observation).reshape(
            1, *self.shapes['observation'])

        self.term[self.cursor] = ar(done).reshape(
            1, *self.shapes['done'])

        #  conditional to reset the counter once we end of the array
        if self.cursor == self.size - 1:
            self.cursor = 0
        else:
            self.cursor += 1

        self.count += 1

    def get_batch(self, batch_size):
        """ randomly samples a batch """
        sample_size = min(batch_size, len(self))
        indicies = np.random.randint(len(self), size=sample_size)

        return {
            'observation': self.obs[indicies],
            'action': self.acts[indicies],
            'reward': self.rews[indicies],
            'next_observation': self.n_obs[indicies],
            'done': self.term[indicies]
        }
