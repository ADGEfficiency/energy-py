# energypy

![Tests](https://github.com/ADGEfficiency/energy-py/actions/workflows/test.yml/badge.svg?branch=main)

A framework for reinforcement learning experiments with energy environments.

## Features

- Electric battery storage environment for energy arbitrage
- Integration with [Gymnasium](https://gymnasium.farama.org/) as a custom Gymnasium environment
- Integration with [Stable Baselines  -> list[dict]3](https://stable-baselines3.readthedocs.io/) for reinforcement learning agents
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

## Example

```shell-session
$ make example
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
