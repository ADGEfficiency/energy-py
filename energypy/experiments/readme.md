## dict experiments

Run using

`$ python dict_expt.py expt_name --run_name --seed`

Experiment is setup in the `dict_expt.py` script

## config file experiments

Run using

`$ python config_expt.py expt_name run_name`

The seed and dataset name are set in the two config files, found at

`energy_py/experiments/results/expt_name/run_configs.ini`

`energy_py/experiments/results/expt_name/common.ini`

The `run_name` argument refers to the section name in `run_configs.ini`
