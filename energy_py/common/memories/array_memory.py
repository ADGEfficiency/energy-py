import numpy as np

from energy_py.common.memories.memory import BaseMemory


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

    def __getitem__(self, idx):
            return (
                self.obs[idx], self.acts[idx], self.rews[idx],
                self.n_obs[idx], self.term[idx]
            )

    def remember(self, observation, action, reward, next_observation, done):
        """ adds experience to the memory """
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
