"""
Runs a single experiment from config files

Command line args
    expt_name - the directory where run results will sit
    dataset_name - name of the dataset folder in experiments/dataset
    --run_name (optional)
    --seed (optional)

Note that here the run_name must be specified, because we need to find the
correct section in run_configs.ini

To run the example experiment
    python config_expt.py example_config example --run_name DEFAULT

Config files are
   experiments/results/expt_name/common.ini
   experiments/results/expt_name/run_configs.ini

When using config files you need to make sure your config args are
being converted into the correct type (int, float etc) after they
are passed into the agent/env

This is because the config files loads the args as strings!
"""

import os

from energy_py import experiment
from energy_py.scripts.experiment import make_expt_parser, run_config_expt

# no logging?

if __name__ == '__main__':
    args = make_expt_parser()

    #  expt path is set here to get the current working directory
    expt_path = os.path.join(os.getcwd(),
                             'results',
                             args.expt_name)

    run_config_expt(args.expt_name, args.run_name, expt_path)
