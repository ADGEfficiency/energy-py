# energypy

![Tests](https://github.com/ADGEfficiency/energy-py/actions/workflows/test.yml/badge.svg?branch=main)

A framework for reinforcement learning experiments with energy environments.

## Features

- Battery storage environments for energy arbitrage
- PPO implementation for training RL agents
- Integration with Gymnasium for custom environments
- Historical electricity price data for realistic training scenarios

## Installation

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

## Usage Examples

Train a PPO agent on the battery storage environment:

```python
from energypy import battery, runner

# Create environment
env = battery.Battery()

# Train agent using PPO
results = runner.train(env, algorithm="PPO")
```

For more examples, see the `examples/` directory.

## License

MIT