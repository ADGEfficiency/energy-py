from energypy.agent.random_policy import RandomPolicy, FixedPolicy
from energypy.agent.memory import Buffer
from energypy.datasets import *
from energypy.envs.battery import Battery
from energypy.envs.gym_wrappers import GymWrapper
from energypy.networks import dense, attention

registry = {
    'lunar': GymWrapper,
    'pendulum': GymWrapper,
    'battery': Battery,
    'random-dataset': RandomDataset,
    'random-policy': RandomPolicy,
    'fixed-policy': FixedPolicy,
    'nem-dataset-dense': NEMDataset,
    'nem-dataset-attention': NEMDatasetAttention,
    'buffer': Buffer,
    'attention': attention,
    # 'dense': Dense,
}


def make(name=None, *args, **kwargs):
    if name is None:
        name = kwargs['name']
    return registry[name](*args, **kwargs)
