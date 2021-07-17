# energy-py

[![Build Status](https://travis-ci.org/ADGEfficiency/energy-py.svg?branch=master)](https://travis-ci.org/ADGEfficiency/energy-py)

energy-py is a framework for running reinforcement learning experiments on energy environments.

The library is focused on electric battery storage, and offers a implementation of a many batteries operating in parallel.

energy-py includes an implementation of the Soft Actor-Critic reinforcement learning agent, implementated in Tensorflow 2:

- test & train episodes based on historical Australian electricity price data,
- checkpoints & restarts,
- logging in Tensorboard.

energy-py is built and maintained by Adam Green - adam.green@adgefficiency.com.


## Setup

```bash
$ make setup
```


## Test

```bash
$ make test
```


## Running experiments

`energypy` has a high level API to run a specific run of an experiment from a `JSON` config file.

The most interesting experiment is to run battery storage for price arbitrage in the Australian electricity market.  This requires grabbing some data from S3.  The command below will download a pre-made dataset and unzip it to `./dataset`:

```bash
$ make pulls3-dataset
```

You can then run the experiment from a JSON file:

```bash
$ energypy benchmarks/nem-battery.json
```

Results are saved into `./experiments/{env_name}/{run_name}`:

```bash
$ tree -L 3 experiments
experiments/
└── battery
    ├── nine
    │   ├── checkpoints
    │   ├── hyperparameters.json
    │   ├── logs
    │   └── tensorboard
    └── random.pkl
```

Also provide wrappers around two `gym` environments - Pendulum and Lunar Lander:

```bash
$ energypy benchmarks/pendulum.json
```

Running the Lunar Lander experiment has a dependency on Swig and pybox2d - which can require a bit of elbow-grease to setup depending on your environment.

```bash
$ energypy benchmarks/lunar.json
```
