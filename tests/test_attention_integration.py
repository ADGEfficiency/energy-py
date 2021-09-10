import energypy

from energypy.datasets import AbstractDataset
import numpy as np


class RandomDatasetAttention(AbstractDataset):
    def __init__(
        self, n=1000, n_features=3, sequence_length=10, n_batteries=2, logger=None
    ):
        super(RandomDatasetAttention, self).__init__()
        self.episode = self.make_random_dataset(
            n, n_features, sequence_length, n_batteries
        )
        self.test_done = True
        self.n_batteries = n_batteries

    def make_random_dataset(self, n, n_features, sequence_length, n_batteries):
        np.random.seed(42)
        #  (timestep, sequence_length, batteries, features)
        prices = np.random.uniform(0, 100, n * n_batteries).reshape(n, n_batteries, 1)
        size = n * sequence_length * n_features * n_batteries
        features = np.random.uniform(0, 100, size).reshape(
            n, n_batteries, sequence_length, n_features
        )

        size = n * sequence_length * sequence_length * n_batteries
        mask = (
            np.random.randint(0, 1, size)
            .reshape(n, n_batteries, sequence_length, sequence_length)
            .astype(bool)
        )
        return {"prices": prices, "features": features, "mask": mask}

    def reset_train(self):
        return self.sample_observation(0)


episode_length = 4
n_batteries = 3
# dataset = energypy.make('random-dataset-attention')
dataset = RandomDatasetAttention(n=10, n_batteries=n_batteries)

env = energypy.make(
    "battery", n_batteries=n_batteries, dataset=dataset, episode_length=episode_length
)

from energypy.sampling import episode

#  run an episode with random policy
from collections import defaultdict

actor = energypy.make("random-policy", env=env)
buffer = energypy.make("buffer", elements=env.elements, size=5)
rewards, infos = episode(
    env,
    buffer,
    actor,
    {"reward-scale": 1},
    counters=defaultdict(int),
    mode="train",
    return_info=True,
)
#  check buffer etc TODO
#  check last row not filled etc

#  try to test with proper agent, train w agent

from energypy.train import train_one_head_network
from energypy.init import init_fresh

hyp = {
    "run-name": "integration",
    "initial-log-alpha": 0.0,
    "gamma": 0.99,
    "rho": 0.995,
    "buffer-size": 100,
    "reward-scale": 500,
    "lr": 3e-4,
    "lr-alpha": 3e-5,
    "batch-size": 4,
    "n-episodes": 4,
    "test-every": 128,
    "n-tests": "all",
    "env": {
        "name": "battery",
        "initial_charge": 0.0,
        "episode_length": episode_length,
        "n_batteries": n_batteries,
        "dataset": dataset,
    },
    "network": {"name": "attention", "size_scale": 8},
    "seed": 42,
}

expt = init_fresh(hyp)

train_one_head_network(
    buffer.sample(4),
    expt["nets"]["actor"],
    [expt["nets"]["online-1"], expt["nets"]["online-2"]],
    [expt["nets"]["target-1"], expt["nets"]["target-2"]],
    expt["nets"]["alpha"],
    expt["writers"],
    expt["optimizers"],
    expt["counters"],
    hyp,
)
