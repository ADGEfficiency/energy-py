# energy_py

**reinforcement learning for energy systems**

The aim of energy_py is to support work on using reinforcement learning for energy problems.  This library provides agents and environments, as well as tools to run experiments.

energy_py is built and maintained by Adam Green - [adam.green@adgefficiency.com](adam.green@adgefficiency.com).  
- [introductory blog post](http://adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/)
- [DQN debugging](http://adgefficiency.com/dqn-debugging/)
- [DDQN hyperparameter tuning](http://adgefficiency.com/dqn-tuning/)
- [introductory Jupyter notebook](https://github.com/ADGEfficiency/energy_py/blob/master/notebooks/examples/Q_learning_battery.ipynb)

## Basic usage

```python
import energy_py

env = energy_py.make_env(env_id='battery')

agent = energy_py.make_agent(
    agent_id='dqn',
    env=env,
    total_steps=1000000
    )

observation = env.reset()

while not done:
    action = agent.act(observation)
    next_observation, reward, done, info = env.step(action)
    training_info = agent.learn()
    observation = next_observation
```

The most common access point for a user will be to run an experiment.  An experiment is run by passing the experiment name and run name as arguments

```bash

cd energy_py/experiments

python experiment.py example dqn

```

Results for this run are then available at

``` bash
cd energy_py/experiments/results/example/dqn
```

The progress of an experiment can be watched with TensorBoard

```bash

tensorboard --logdir='./energy_py/experiments/results'

```

![fig](assets/tb1.png)

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

$ pip install --ignore-installed -r requirements.txt

```
## Project 

The aim of energy_py is to provide 
- high quality implementations of agents suited to solving energy problems
- mutiple energy environments
- tools to run experiments

The design philosophies of energy_py
- simple class heirarchy structure (maximum of two levels (i.e. parent child)
- utilize Python standard library (deques, namedtuples etc) where possible
- utilize TensorFlow & TensorBoard
- provide sensible defaults for args

### Agents

energy_py is currently focused on a high quality impelementation of DQN and implementations of naive and heuristic agents for comparison.

DQN was chosen because it is
- established algorithm,
- many examples of DQN implementations,
- highly extensible (DDQN, prioritized experience replay, dueling, n-step returns - see [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) for a summary
- most energy environments have low dimensional action spaces (making discretization tractable).  Discretization still means a loss of action space shape, but the action space dimensionality is reasonable.

Naive agents include an agent that randomly samples the action space, independent of observation.  Heuristic agents are
usually custom built for a specific environment.  Examples of heuristic agents include actions based on the time of day or on the values of a forecast.

### Environments

energy_py provides custom built models of energy environments and wraps around Open AI gym.  Support for basic gym
models is included to allow debugging of agents with familiar environments.

#### gym environments

- CartPole-v0 - [gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) - [energy_py](https://github.com/ADGEfficiency/energy_py/blob/dev/energy_py/envs/register.py)

- Pendulum-v0 - [gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py) - [energy_py](https://github.com/ADGEfficiency/energy_py/blob/dev/energy_py/envs/register.py)

- MountainCar-V0' - [gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py) - [energy_py](https://github.com/ADGEfficiency/energy_py/blob/dev/energy_py/envs/register.py)

#### energy_py environments

- [electric battery storage](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.p://github.com/ADGEfficiency/energy_py/tree/dev/energy_py/envs/battery)

Dispatch of a battery arbitraging wholesale prices.  

Battery is defined by a capacity and a maximum rate to charge and discharge, with a round trip efficieny applied on storage.

- [demand side flexibility](https://github.com/ADGEfficiency/energy_py/tree/dev/energy_py/envs/flex)

Dispatch of price responsive demand side flexibility.  Flexible assset is a chiller system, with an action space of the return temperature setpoint.
