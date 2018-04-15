
# energy_py

**energy_py is reinforcement learning for energy systems**

Using reinforcement learning agents to control virtual energy environments is a necessary step in using reinforcement learning to optimize real world energy systems.

energy_py supports this goal by providing a **collection of agents, energy environments and tools to run experiments.**

energy_py is built and maintained by Adam Green - [adam.green@adgefficiency.com](adam.green@adgefficiency.com).  Read more about the motivations and design choics of the project on the [introductory blog post](http://adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/).

## Basic usage


Environments and agents can be created using a low-level API similar to OpenAI gym.

```python
import energy_py

TOTAL_STEPS = 1000

env = energy_py.make_env(env_id='BatteryEnv',
                         dataset_name=example,
                         episode_length=288,
                         power_rating=2}

agent = energy_py.make_agent(agent_id='DQN',
                             env=env
                             total_steps=TOTAL_STEPS)

observation = env.reset()

action = agent.act(observation)

next_observation, reward, done, info = env.step(action)

training_info = agent.learn()

```
A detailed example of the low level energy_py framework is given in a Jupyter Notebook using the [DQN agent with the Battery environment](https://github.com/ADGEfficiency/energy_py/blob/master/notebooks/examples/Q_learning_battery.ipynb).

The higher level energy_py API allows running of experiments from [config dictionaries](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/experiments/dict_expt.py) or from [config.ini files](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/experiments/config_expt.py).

Single call using the experiment function

```python
energy_py.experiment(agent_config,
                     env_config,
                     total_steps,
                     paths=energy_py.make_paths('path/to/results')
```
Running a config dictionary experiment from a Terminal.  The experiment will be called 'example_expt' and will use the
'example' dataset.

```bash
$ cd energy_py/energy_py/experiments

$ python config_expt.py example_expt example  
```

## Installation

To install energy_py using Anaconda

```bash
$ conda create --name energy_py python=3.5.2

$ activate energy_py (windows)
or
$ source activate energy_py (unix)

$ git clone https://github.com/ADGEfficiency/energy_py.git

$ cd energy_py

$ python setup.py install (using package)
or
$ python setup.py develop (developing package)

$ pip install requirements.txt

```
## Project structure

The aim of energy_py is to provide 
- one high quality implementation of DQN and it's extensions
- mutiple energy environments
- tools to run experiments

### Agents
The reason for choosing to implement only DQN (and not mutiple different agents such as DPG, A3C etc) is that so far all
the energy_py environments have low dimensional action spaces.  The large number of extensions to DQN (DDQN, prioritized
experience replay, dueling architecture etc) mean that implementing DQN with these extensions should enable a high
quality agent. 

A good summary of DQN variants is given in [Hessel et. al (2017) Rainbow: Combining Improvements in Deep Reinforcement
Learning](https://arxiv.org/pdf/1710.02298.pdf).
- DQN - target network & experience replay
- prioritized experience replay
- DDQN
- dueling architecture

Also implemented are simpler agents such as RandomAgent or agents based on determinsitic rules (usually handcrafted for
a specific environment).

### Environments
The unique contrbition of energy_py are energy focused environments.  Currently implemented environments:

**Battery storage**

env_id = 'BatteryEnv'

Model of a electric battery.  Optimal dispatch of a battery arbitraging wholesale prices.

**Flex-v0**

env_id = 'Flex-v0'

Model of a flexibility (i.e. demand side response) asset.  Agent can operate two cycles.  Cycle is a fixed length.
1. flex_up/flex_down/relax
2. flex_down/flex_up/relax

**Flex-v1**

env_id = 'Flex-v1'

Agent can operate a flex_down/flex_up/relax cycle.  Agent can choose to stop the flex_down period.

### Tools to run experiments
In addition to the agents and environments energy_py also provides tools to run experiments.  Visualization of experiment results is done using TensorBoard.
