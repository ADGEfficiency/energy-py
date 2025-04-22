# energypy

![Tests](https://github.com/ADGEfficiency/energy-py/actions/workflows/test.yml/badge.svg?branch=main)

A framework for reinforcement learning experiments with energy environments.

## Features

- Electric battery storage environment for energy arbitrage
- Integration with [Gymnasium](https://gymnasium.farama.org/) as a custom Gymnasium environment
- Integration with [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) for reinforcement learning agents
- Historical electricity price data for realistic training scenarios
- Experiment framework for training and evaluation on separate datasets
- Tensorboard logging for experiment tracking

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

## Examples

```shell-session
$ python examples/battery.py
```

Or run a more extensive experiment with real electricity price data:

```shell-session
$ python examples/battery_arbitrage_experiments.py
```

## Usage

### Basic Battery Environment

```python
import gymnasium as gym
import energypy

# Register the environment
env_id = "energypy/battery"
gym.register(
    id=env_id,
    entry_point="energypy:Battery",
)

# Create the environment with custom parameters
env = gym.make(env_id, electricity_prices=prices)
env = gym.wrappers.NormalizeReward(env)
```

### Training with PPO

```python
from stable_baselines3 import PPO
import energypy

# Configure the experiment
config = energypy.ExperimentConfig(
    env_tr=env,
    agent=PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./data/tensorboard",
    ),
    name="battery",
    n_learning_steps=50000,
    n_eval_episodes=10,
)

# Run the experiment
result = energypy.run_experiment(cfg=config)
```

### Running Multiple Experiments

```python
import energypy

# Define multiple experiment configurations
configs = []
for param in parameter_values:
    config = energypy.ExperimentConfig(...)
    configs.append(config)

# Run all experiments
results = energypy.run_experiments(
    configs, log_dir="./data/tensorboard/experiment_name"
)
```

### Monitoring Results

View experiment logs with Tensorboard:

```shell-session
$ tensorboard --logdir ./data/tensorboard/
```
