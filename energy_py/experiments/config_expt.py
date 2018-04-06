"""
Runs a single experiment from config files

Command line args
    expt_name - the directory where run results will sit
    dataset_name - name of the dataset folder in experiments/dataset
    --run_name
    --seed (optional)

Note that here the run_name must be specified, because we need to find the
correct section in run_configs.ini

To run the example experiment
    python config_expt.py example_config example --run_name DQN_1

Config files are
   experiments/results/expt_name/common.ini
   experiments/results/expt_name/run_configs.ini
"""

import os

from energy_py import make_expt_parser, run_config_expt


if __name__ == '__main__':
    args = make_expt_parser()

    #  expt path is set here to get the current working directory
    expt_path = os.path.join(os.getcwd(),
                             'results',
                             args.expt_name)

    run_config_expt(args.expt_name, args.run_name, expt_path)
