import random

import numpy as np

from energy_py import Experience, SumTree, MinTree, calculate_returns
from energy_py.agents import PrioritizedReplay


def generate_experience(obs_shape, action_shape):
    """
    A function to generate namedtuples of experience
    """
    obs = np.random.rand(1, *obs_shape)
    act = np.random.rand(1, *action_shape)
    rew = np.random.randint(10)
    next_obs = np.random.rand(1, *obs_shape)
    terminal = False

    return Experience(obs, act, rew, next_obs, terminal)


def setup_memory(size, num_exps, alpha=0.7):
    obs_shape, action_shape = (4,), (2,)
    mem = PrioritizedReplay(10, obs_shape, action_shape, alpha)
    exps = [generate_experience(obs_shape, action_shape) for _ in range(5)]
    return mem, exps


def test_calc_returns():
    rews = [10, 15, -4, 8, -1]
    discount = 1
    rtn = -1 + discount*8 + discount**2*-4 + discount**3*15 + discount**4*10

    test_rtn = calculate_returns(rews, discount)

    assert test_rtn[0] == rtn


def test_remember():
    """
    Checks the priorities are stored correctly
    """
    mem, exps = setup_memory(10, 5)
    pr = [random.random() for _ in range(len(exps))]

    #  code relies on the memory size being longer than the exps
    assert len(exps) < mem.size

    for e, p in zip(exps, pr):
        print(e)
        mem.remember(*e._asdict(), priority=p)

    for idx, e in enumerate(mem.experiences):
        p = mem.sumtree[idx]
        expected = pr[idx]

        assert p == expected


def test_trees():
    """
    Tests the sum and min operations over the memory
    """
    mem, exps = setup_memory(10, 5)

    pr = [random.random() for _ in range(len(exps))]

    for e, p in zip(exps, pr):
        mem.remember(*e._asdict(), priority=p)

    sumtree = mem.sumtree
    mintree = mem.mintree

    s1 = sumtree.sum()
    m1 = mintree.min()

    tol = 1e-6
    assert s1 - sum(pr[-10:]) < tol
    assert m1 - min(pr[-10:]) < tol


def test_update_priorities():
    mem, exps = setup_memory(10, 5, alpha=1.0)

    for exp in exps:
        #  remember experience using the default
        mem.remember(*exp._asdict().values())

    assert mem.sumtree.sum() == 5

    #  get a batch
    batch = mem.get_batch(2, beta=1)

    td_errors = np.array([0.1, 100]).reshape(2, 1)
    indicies = batch['indexes']
    mem.update_priorities(indicies, td_errors)

    np.testing.assert_allclose(mem.mintree.min(), 0.1, rtol=1e-3)
    np.testing.assert_allclose(mem.sumtree.sum(), 100+3+0.1, rtol=1e-3)

if __name__ == '__main__':
    test_remember()
    test_trees()
