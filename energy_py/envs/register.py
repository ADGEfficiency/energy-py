"""
A registery for environments supported by energy_py

Combination of native energy_py environments and wrapped gym environments
"""

import logging

from energy_py.envs.flex import Flex
from energy_py.envs.battery import Battery
from energy_py.envs.gym import CartPoleEnv, PendulumEnv, MountainCarEnv


logger = logging.getLogger(__name__)

env_register = {'flex': Flex,
                'battery': Battery,
                'cartpole-v0': CartPoleEnv,
                'pendulum-v0': PendulumEnv,
                'mountaincar-v0': MountainCarEnv}


def make_env(env_id, **kwargs):
    logger.info('Making env {}'.format(env_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    env = env_register[env_id]

    return env(**kwargs)
