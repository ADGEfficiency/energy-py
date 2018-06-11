import argparse
import datetime
import os

from energy_py.common import ensure_dir


def make_expt_parser():
    """
    Parses arguments from the command line for running experiments

    returns
        args (argparse NameSpace)
    """
    parser = argparse.ArgumentParser(description='energy_py experiment argparser')

    #  required
    parser.add_argument('expt_name', default=None, type=str)
    parser.add_argument('dataset', default=None, type=str)
    #  optional
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()

    return args


def make_config_parser():
    """
    Parses arguments from the command line for running config experiments

    returns
        args (argparse NameSpace)
    """
    parser = argparse.ArgumentParser(description='energy_py experiment argparser')

    #  required
    parser.add_argument('expt_name', default=None, type=str)
    parser.add_argument('run_name', default=None, type=str)

    args = parser.parse_args()

    return args


def make_paths(expt_path, run_name=None):
    """
    Creates a dictionary of paths for use with experiments

    args
        expt_path (str)
        run_name (str) optional name for run.  Timestamp used if not given

    returns
        paths (dict) {name: path}

    Folder structure
        experiments/results/expt_name/run_name/tensoboard/run_name/rl
                                                                  /act
                                                                  /learn
                                               env_histories/ep_1/hist.csv
                                                             ep_2/hist.csv
                                                             e..
                                               common.ini
                                               run_configs.ini
                                               agent_args.txt
                                               env_args.txt
                                               info.log
                                               debug.log
    """
    #  use a timestamp if no run_name is supplied
    if run_name is None:
        run_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    #  rename the join function to make code below eaiser to read
    join = os.path.join

    #  run_path is the folder where output from this run will be saved in
    run_path = join(expt_path, run_name)

    paths = {'run_path': run_path,

             #  config files
             'common_config': join(expt_path, 'common.ini'),
             'run_configs': join(expt_path, 'run_configs.ini'),

             #  tensorboard runs are all in the tensoboard folder
             #  this is for easy comparision of run
             'tb_rl': join(expt_path, 'tensorboard', run_name, 'rl'),
             'tb_act': join(expt_path, 'tensorboard', run_name, 'act'),
             'tb_learn': join(expt_path, 'tensorboard', run_name,  'learn'),
             'env_histories': join(run_path, 'env_histories'),

             #  run specific folders are in another folder
             'debug_log': join(run_path, 'debug.log'),
             'info_log': join(run_path, 'info.log'),
             'env_args': join(run_path, 'env_args.txt'),
             'agent_args': join(run_path, 'agent_args.txt'),
             'ep_rewards': join(run_path, 'ep_rewards.csv')}

    #  check that all our paths exist
    for key, path in paths.items():
        ensure_dir(path)

    return paths
