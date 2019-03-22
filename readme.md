# energy-py

[![Build Status](https://travis-ci.org/ADGEfficiency/energy-py.svg?branch=master)](https://travis-ci.org/ADGEfficiency/energy-py)

energypy is a framework for running reinforcement learning experiments on energy environments.  energypy is built and maintained by Adam Green - [adam.green@adgefficiency.com](adam.green@adgefficiency.com).

## Installation

```bash
$ git clone https://github.com/ADGEfficiency/energy-py

$ pip install --ignore-installed -r requirements.txt

$ python setup.py install
```

## Basic use

The most common access point for a user will be to run an experiment.  An experiment is run by passing a `yaml` config file along with the name of the run:

```bash
$ ep-experiment energypy/experiments/example_config.yaml run1
```

An example config file (`energypy/experiments/example_config.yaml`):

```yaml
expt:
    name: test

run1: &defaults
    total_steps: 10000

    env:
        env_id: battery
        dataset: example

    agent:
        agent_id: random

run2:
    <<: *defaults
    agent:
        agent_id: dqn
```

Results (log files for each episode & experiment summaries) are placed into a folder in the users `$HOME`.  The progress of an experiment can be watched with TensorBoard by running a server looking at this results folder:

```bash
$ tensorboard --logdir='~/energy-py-results'
```

## Low level API

energypy provides the familiar gym style low-level API for agent and environment initialization and interactions:

```python
import energypy

env = energypy.make_env(env_id='battery')

agent = energypy.make_agent(
    agent_id='dqn',
    env=env,
    total_steps=10000
	)

observation = env.reset()

while not done:
    action = agent.act(observation)
    next_observation, reward, done, info = env.step(action)
    training_info = agent.learn()
    observation = next_observation
```

## Library

energy-py currently implements:

- naive agents
- DQN agent
- Battery storage environment
- Demand side flexibility environment
- Wrappers around the OpenAI gym CartPole, Pendulum and MountainCar environments

## Further reading

- [introductory blog post](http://www.adgefficiency.com/energypy-reinforcement-learning-for-energy-systems/)
- [DQN debugging](http://adgefficiency.com/dqn-debugging/)
- [DDQN hyperparameter tuning](http://adgefficiency.com/dqn-tuning/)
- [Jupyter notebook example of low level API - DQN and battery environment](https://github.com/ADGEfficiency/energypy/blob/master/notebooks/examples/DQN_battery_example.ipynb)
- [talk covering two years of lessons working on energypy](https://gitpitch.com/ADGEfficiency/energy-py-talk#/)
