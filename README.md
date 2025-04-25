# energypy

![Tests](https://github.com/ADGEfficiency/energy-py/actions/workflows/test.yml/badge.svg?branch=main)

A framework for running reinforcement learning experiments on energy environments - starting with electric battery storage.

## Features

- Electric battery storage environment for energy arbitrage
- Integration with [Gymnasium](https://gymnasium.farama.org/) as a custom Gymnasium environment
- Integration with [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) for reinforcement learning agents
- Historical electricity price data for realistic training scenarios
- Experiment framework for training and evaluation on separate datasets
- Tensorboard logging for experiment tracking

## Setup

```shell-session
$ make setup
```

## Examples

```shell-session
$ uv run examples/battery.py
```

Or run a more extensive experiment with real electricity price data:

```shell-session
$ uv run examples/battery_arbitrage_experiments.py
```

## Test

```shell-session
$ make test
```
