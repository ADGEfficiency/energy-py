# energypy

![Tests](https://github.com/ADGEfficiency/energy-py/actions/workflows/test.yml/badge.svg?branch=main)

A framework for reinforcement learning experiments with energy environments.

## Features

- Battery storage environments for energy arbitrage
- PPO implementation for training RL agents
- Integration with [Gymnasium](https://gymnasium.farama.org/) for custom environments
- Integration with [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for access to state-of-the-art RL algorithms
- Historical electricity price data for realistic training scenarios

## Setup

```bash
make setup
```

## Development

```bash
# Run tests
make test

# Check code style and linting
make check

# Verify type annotations
make static
```

## Usage

Train a PPO agent on the battery storage environment:

```python
import energypy

env = energypy.Battery()
results = energypy.train(env, "PPO", name="battery")
```

Experiment logs to Tensorboard:

```shell-session
$ tensorboard --logdir ./data/tensorboard/
```
