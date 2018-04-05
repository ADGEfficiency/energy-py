"""
Known issues - the episode_random argument from the parse ini
"""
import configparser
import os

import energy_py

from energy_py import make_paths, make_logger


def parse_ini(filepath, section):
    config = configparser.ConfigParser()
    config.read(filepath)
    return dict(config[section])


if __name__ == '__main__':

    base_path = os.path.join(os.getcwd(),
                             'results',
                             'config_test')

    paths = make_paths(base_path)
    logger = make_logger(paths, 'master')

    common = parse_ini(paths['common'], 'experiment')

    env_config = parse_ini(paths['common'], 'env')

    env = energy_py.make(**env_config)
