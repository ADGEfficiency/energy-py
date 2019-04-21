""" checking episode sample strageties """

import numpy as np
import pytest

import energypy


@pytest.mark.parametrize(
    'env, sample_strat, episode_length',
    (('battery', 'random', 32),
     ('battery', 'random', 64),
     ('battery', 'fixed', 256),
     ('battery', 'fixed', 128),
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
        if sample_strat == 'full':
            assert len(data) == env.state_space.num_samples

        else:
            assert len(data) == episode_length

    next_states = np.array(i['next_state']).reshape(-1, *env.state_space.shape)

    # assert next_states[-1] == np.zeros((1, *env.state_space.shape))


if __name__ == '__main__':
    env = 'battery'
    sample_strat = 'full'
    episode_length = 4

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
        if sample_strat == 'full':
            assert len(data) == env.state_space.num_samples

        else:
            assert len(data) == episode_length

    next_states = np.array(i['next_state']).reshape(-1, *env.state_space.shape)

    np.testing.assert_array_equal(
        next_states[-1], np.zeros((*env.state_space.shape))
    )

    next_observations = np.array(i['next_observation']).reshape(-1, *env.observation_space.shape)

    np.testing.assert_array_equal(
        next_observations[-1], np.zeros((*env.observation_space.shape))
    )

