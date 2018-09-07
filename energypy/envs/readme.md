# environments

energy_py provides environments that energy time series problems, focusing on battery storage and flexibility.  Also provided are wrappers around Open AI gym environments such as cartpole and mountain-car.

Environments are accessed via a [register](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/register.py)

## energy_py energy environments

Data for the time series is loaded from a `state.csv` and `observation.csv` 

These csvs are found in the relevant dataset folder

The example dataset is in `energy_py/experiments/datasets/example`

All of energy_py works on a **5 minute basis**  

## Open AI gym environments
Custom built wrappers are made around gym environments to allow use with energy_py agents via the same API as for energy_py envs

gym environments are included because they allow benchmarking of agents on well built and formulated environments
