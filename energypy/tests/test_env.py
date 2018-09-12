""" checking episode sample strageties """

import pandas as pd

import energypy


def test_random():
    env = energypy.make_env(
        'flex',
        episode_sample='random',
        episode_length=24
    )

    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        s, r, done, i = env.step(action)

    i = pd.DataFrame().from_dict(i)

    assert i.shape[0] == 24


def test_full():
    env = energypy.make_env(
        'flex',
        episode_sample='full'
    )

    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        s, r, done, i = env.step(action)

    i = pd.DataFrame().from_dict(i)

    assert i.shape[0] == env.state_space.data.shape[0]

def test_fixed():
    env = energypy.make_env(
        'flex',
        episode_sample='fixed',
        episode_length=108
    )

    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        s, r, done, i = env.step(action)

    i = pd.DataFrame().from_dict(i)

    assert i.shape[0] == 108

if __name__ == '__main__':
    test_fixed()
