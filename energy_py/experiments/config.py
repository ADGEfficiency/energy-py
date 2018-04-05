"""
test - can test by reading the log file!
"""
import configparser
import os

import energy_py

from energy_py import make_paths, make_logger, experiment


def parse_ini(filepath, section):
    """
    args
        filepath (str) location of the .ini
        section (str) section of the ini to read

    returns
        config_dict (dict)
    """
    logger.info('reading {}'.format(filepath))
    config = configparser.ConfigParser()
    config.read(filepath)

    #  check to convert boolean strings to real booleans
    config_dict = dict(config[section])

    for k, val in config_dict.items():
        if val == 'True':
            config_dict[k] = True

        if val == 'False':
            config_dict[k] = False

    return config_dict


if __name__ == '__main__':

    #  entered in from the command line
    expt_name = 'config_test'
    run_name = 'DQN_1'

    run_path = os.path.join(os.getcwd(),
                            'results',
                            expt_name)

    paths = make_paths(run_path, run_name=run_name)
    logger = make_logger(paths, 'master')

    env_config = parse_ini(paths['common_config'], 'env')
    agent_config = parse_ini(paths['run_configs'], run_name)

    results = experiment(agent_config,
                         env_config,
                         agent_config['total_steps'],
                         paths,
                         seed=agent_config['seed'])
