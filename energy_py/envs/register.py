""" register for energy_py and gym envs """

import logging

from energy_py.envs.flex import Flex
from energy_py.envs.battery import Battery
from energy_py.envs.gym import CartPoleEnv, PendulumEnv, MountainCarEnv

from energy_py.envs.twenty_forty_eight.ep_wrapper import Game2048


logger = logging.getLogger(__name__)

env_register = {
    'flex': Flex,
    'battery': Battery,
    'cartpole-v0': CartPoleEnv,
    'pendulum-v0': PendulumEnv,
    'mountaincar-v0': MountainCarEnv,
    '2048': Game2048
}


def make_env(env_id, **kwargs):
    logger.info('Making env {}'.format(env_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    env = env_register[str(env_id)]

    return env(**kwargs)
