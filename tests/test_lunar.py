import pytest

from energypy.agent.random_policy import make as make_random_policy
from energypy.envs.gym_wrappers import GymWrapper


def test_pendulum():
    env = GymWrapper('pendulum')
    policy = make_random_policy(env)


@pytest.mark.pybox2d
def test_lunar():
    env = GymWrapper('lunar')
    policy = make_random_policy(env)
