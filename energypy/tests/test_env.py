""" checking episode sample strageties """

import pandas as pd

import energypy


def random(env):
    env = energypy.make_env(
        env,
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


def full(env):
    env = energypy.make_env(
        env,
        episode_sample='full'
    )

    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        s, r, done, i = env.step(action)

    i = pd.DataFrame().from_dict(i)

    assert i.shape[0] == env.state_space.data.shape[0]


def fixed(env):
    env = energypy.make_env(
        env,
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


tests = [random, full, fixed]


def test_battery():
    [test('battery') for test in tests]


def test_flex():
    [test('flex') for test in tests]
