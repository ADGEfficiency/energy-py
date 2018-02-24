from collections import namedtuple

from energy_py import calculate_returns


Experience = namedtuple('Experience', ['observation',
                                       'action',
                                       'reward',
                                       'next_observation',
                                       'terminal',
                                       'priority'])

def generate_experience(p=None):
    """
    A function to generate namedtuples of experience
    """
    obs = np.random.rand(1, 4)
    act = np.random.rand(1, 2)
    rew = np.random.randint(10)
    next_obs = np.random.rand(1, 2)
    terminal = False

    if p is None:
        prio = random.random()
    else:
        prio = p

    return Experience(obs, act, rew, next_obs, terminal, prio)

def test_calc_returns():
    rews = [10, 15, -4, 8, -1]
    discount = 1
    rtn = -1 + discount*8 + discount**2*-4 + discount**3*15 + discount**4*10

    test_rtn = calculate_returns(rews, discount)

    assert test_rtn[0] == rtn
