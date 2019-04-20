""" checking episode sample strageties """

import energypy

import pytest


@pytest.mark.parametrize(
    'env, episode_sample, episode_length',
    (('battery', 'random', 32),
     ('battery', 'fixed', 2016),
     ('battery', 'full', 0),
     ('flex', 'random', 32),
     ('flex', 'fixed', 2016),
     ('flex', 'full', 0))
)
def test_env_lengths(env, episode_sample, episode_length):
    env = energypy.make_env(
        env_id=env,
        episode_sample=episode_sample,
        episode_length=episode_length
    )

    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        s, r, done, i = env.step(action)

    for key, data in i.items():
        if episode_sample == 'full':
            assert len(data) == env.state_space.data.shape[0]

        else:
            assert len(data) == episode_length
