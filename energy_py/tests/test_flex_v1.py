"""
A suite of tests for the FlexV1 environment
"""
import numpy as np

import energy_py


env_config = {'dataset': 'example',
              'env_id': 'flex-v1',
              'flex_size': 2,
              'max_flex_time': 4,
              'relax_time': 4}

env = energy_py.make_env(**env_config)

#  pull out the prices so that we can calculate rewards by hand
PRICES = env.env.state_ts.values

def test_flex_once():
    """
    Flex once straight away and let cycle finish
    """
    o = env.reset()
    for step in range(20):

        if step == 0:
            o, r, d, i = env.step(np.array(1))
        else:
            o, r, d, i = env.step(np.array(0))

    flex_profile = np.concatenate([np.full(shape=4, fill_value=2),
                                   np.full(shape=4, fill_value=-2)])
    p = PRICES[:flex_profile.shape[0]]
    rews = np.sum(p.flatten() * flex_profile.flatten())/12

    info_rews = sum(i['reward'])
    assert np.isclose(rews, info_rews)


def test_flex_start_stop():
    """
    Flex once after two steps, then stop after 2 steps
    """
    o = env.reset()
    for step in range(20):
        #  default action of doing nothing
        action = np.array(0)
        if step == 1:
            action = np.array(1)
        if step == 3:
            action = np.array(2)

        o, r, d, i = env.step(action)


    flex_profile = np.concatenate([np.array([0]),
                                   np.full(shape=2, fill_value=2),
                                   np.full(shape=2, fill_value=-2)])

    p = PRICES[:flex_profile.shape[0]]
    rews = np.sum(p.flatten() * flex_profile.flatten())/12

    info_rews = sum(i['reward'])
    assert np.isclose(rews, info_rews)


def test_no_op():
    """
    Testing that sending action=0 or 2 do nothing
    """
    o = env.reset()
    for step in range(20):
        if step % 2 == 0:
            action = np.array(0)
        else:
            action = np.array(2)

        o, r, d, i = env.step(action)

    rews = np.sum(i['rews'])
    assert rews == 0
