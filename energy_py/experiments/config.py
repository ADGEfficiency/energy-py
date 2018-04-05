"""
test - can test by reading the log file!
"""
import os

from energy_py import make_expt_parser, run_config_expt


if __name__ == '__main__':

    #  entered in from the command line
    expt_name = 'config_test'
    run_name = 'DQN_1'

    #  expt path is set here to get the current working directory
    expt_path = os.path.join(os.getcwd(),
                             'results',
                             expt_name)

    results = run_config_expt(expt_name, run_name, expt_path)
