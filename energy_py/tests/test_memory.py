from collections import namedtuple
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

