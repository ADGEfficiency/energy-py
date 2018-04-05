"""
Known issues - the episode_random argument from the parse ini
double Q
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

    #  entered in from the command line
    expt_name = 'config_test'

    base_path = os.path.join(os.getcwd(),
                             'results',
                             expt_name)

    paths = make_paths(base_path)
    logger = make_logger(paths, 'master')

    common = parse_ini(paths['common'], 'experiment')
    env_config = parse_ini(paths['common'], 'env')

    env = energy_py.make_env(**env_config)

    #  input from command line
    run_config = os.path.join(os.getcwd(),
                              'results',
                              expt_name,
                              'expt1.ini')

    agent_config = parse_ini(run_config, 'agent')

    agent = energy_py.make_agent(**agent_config)


