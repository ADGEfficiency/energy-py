""" checking episode sample strageties """

import energypy

import pytest


@pytest.mark.parametrize(
    'env, sample_strat, episode_length',
    (('battery', 'random', 32),
     ('battery', 'random', 64),
     ('battery', 'fixed', 2016),
     ('battery', 'full', 0))
)
def test_env_lengths(env, sample_strat, episode_length):
    env = energypy.make_env(
        env_id=env,
        sample_strat=sample_strat,
        episode_length=episode_length
    )

    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        s, r, done, i = env.step(action)

    for key, data in i.items():
        import pdb; pdb.set_trace()
        if sample_strat == 'full':
            assert len(data) == env.state_space.num_samples

        else:
            assert len(data) == episode_length
