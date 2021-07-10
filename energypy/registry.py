from energypy.agent.random_policy import RandomPolicy, FixedPolicy
from energypy.agent.memory import Buffer
from energypy.datasets import *
from energypy.envs.battery import Battery
from energypy.envs.gym_wrappers import GymWrapper


registry = {
    'lunar': GymWrapper,
    'pendulum': GymWrapper,
    'battery': Battery,
    'random-dataset': RandomDataset,
    'random-policy': RandomPolicy,
    'fixed-policy': FixedPolicy,
    'nem-dataset': NEMDataset,
    'buffer': Buffer,
}


def make(name=None, *args, **kwargs):
    if name is None:
        name = kwargs['name']
    return registry[name](*args, **kwargs)
