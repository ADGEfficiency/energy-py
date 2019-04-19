""" register for energypy and gym envs """

# from energypy.envs.flex import Flex
from energypy.envs.battery import Battery
from energypy.envs.gym import CartPoleEnv, PendulumEnv, MountainCarEnv

env_register = {
    # 'flex': Flex,
    'battery': Battery,
    'cartpole-v0': CartPoleEnv,
    'pendulum-v0': PendulumEnv,
    'mountaincar-v0': MountainCarEnv,
}


def make_env(env_id, **kwargs):
    """ grabs class from register and initializes """
    return env_register[str(env_id)](**kwargs)
