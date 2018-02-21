## energy_py

**energy_py is reinforcement learning for energy systems.**

Using reinforcement learning agents to control virtual energy environments is the first step towards using reinforcement learning to optimize real world energy systems.

energy_py supports this goal by providing a **collection of reinforcement learning agents, energy environments and tools to run experiments.**

energy_py is built and maintained by Adam Green.  This project is in rapid development - if you would like to get involved send me an email at [adam.green@adgefficiency.com](adam.green@adgefficiency.com).  I write about energy & machine learning at [adgefficiency.com](http://adgefficiency.com/).  The introductory blog post for this project [is here.](http://adgefficiency.com/energy_py-reinforcement-learning-for-energy-systems/)

I teach a one day [introduction to reinforcement learning learning](https://github.com/ADGEfficiency/DSR_RL) class at [Data Science Retreat](https://www.datascienceretreat.com/).

### Work to be done
Problem with [Normalizer](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/scripts/processors.py) - the object remembers the min & max for each dimension for it's entire lifetime.  High initial values make the scaling silly.

Functionality to [track & log rewards during experiment() should be inside the Runner class.](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/scripts/experiment.py)

Ability to clip rewards.  Most likely should sit in self.remember() in the Memory object before the reward is added to the memory.  

For energy_py environments - the raw_state.csv or state.csv, observation.csv dependency is a bit of an issue.  The idea is that the user will have their own electricity price profile, but I'd like to build in the ability for the env to generate a profile if the user doesn't supply a raw_state.csv or state.csv, observation.csv.  

Remove pandas dependencies.

#### Prioritized experience replay
[Schaul et. al (2015) Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

Some implementations used a binary heap search tree.  The Python standard library has a [collection of functions for heap queues](https://docs.python.org/3/library/heapq.html).

[General intro to binary heaps with some Python implementation from scratch](http://interactivepython.org/runestone/static/pythonds/Trees/BinaryHeapImplementation.html).

[Takoika/PrioritizedExperienceReplay implementation](https://github.com/takoika/PrioritizedExperienceReplay/blob/master/sum_tree.py).

[TensorForce implementation](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/core/memories/prioritized_replay.py)

[Slide 20 of 'Deep Reinforcment Learning in TensorFlow'](http://web.stanford.edu/class/cs20si/lectures/slides_14.pdf) - samples using log-probabilities (not a search tree).

#### Auxillary loss functions
[Raia Hadsell on "Deep Reinforcement Learning and Real World Challenges"](https://www.youtube.com/watch?v=0e_uGa7ic74)

#### Creating a Policy class/function
ie e-greedy, random, soft-max, Boltzmann

[Action-Selection Strategies for Exploration](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf)

[This implementation of DQN](https://ewanlee.github.io/2017/07/09/Using-Tensorflow-and-Deep-Q-Network-Double-DQN-to-Play-Breakout/) defines the policy as a method.

#### Heuristic pre-processing
A way to override action selection using a determinsitic heuristic.  Might be a use case for a decorator.

#### Tests
Currently have a few for testing the TensorFlow implementations and the processors.

### Basic usage

[A simple and detailed example](https://github.com/ADGEfficiency/energy_py/blob/master/notebooks/examples/Q_learning_battery.ipynb) of using the DQN agent to control the battery environment is a great place to start.


Another way to use energy_py is to run experiments.
 The script [gym_expt.py](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/experiments/gym_expt.py) will run Gym experiments.  

The script [ep_expt.py](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/experiments/ep_expt.py) will run energy_py experiments.  The function used to run an experiment is found in [scripts/experiment.py.](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/scripts/experiment.py)

For an environment to be used it must be wrapped in the environment registry in [register.py](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/register.py).  The registry allows consistency in the attributes and methods used by agents.  

### Installation
Below I use Anaconda to create a Python 3.5 virtual environment.  You can of course use your own environment manager.

If you just want to install to your system Python you can skip to cloning the repo.  
```
conda create --name energy_py python=3.5.2
```
Activate the virtual environment
```
activate energy_py (windows)

source activate energy_py (unix)
```
Clone the repo somewhere
```
git clone https://github.com/ADGEfficiency/energy_py.git
```
Move into the energy_py folder and install using setup.py.  This will install energy_py into your activated Python environment
```
cd energy_py
python setup.py install
```
Finally install the required packages
```
pip install requirements.txt
```
The main dependencies of energy_py are numpy, pandas & TensorFlow.  

### Project structure

Environments are created by inheriting from the [BaseEnv](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/env_core.py) class.

Agents are created by inheriting from the [BaseAgent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/agent.py) class.  

Agents use [Memory](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/memory.py) objects to remember and sample experience.  

The logging module is used to log INFO to console and DEBUG to file.  TensorBoard is used to track data during acting, learning and for RL specific stuff like rewards.

**Environments**

Agent and environment interaction is shown below - it follows the standard
Open AI gym API for environments i.e. .reset, .step(action).

```
from energy_py.agents import DQN, tfValueFunction
from energy_py.envs import BatteryEnv

env = BatteryEnv()

agent = DQN(env,
            discount=0.9,
            Q=tfValueFunction,
            batch_size=64,
            total_steps=1000)

obs = env.reset()
action = agent.act(observation=obs)
next_obs, reward, done, info = env.step(action)
agent.memory.add_experience(obs, action, reward, next_obs, step, episode)

```
The following environments are implemented:

- [Electricity storage in a battery](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/battery)

- [Generic flexibility action environment](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/flex)

In energy_py v1.0 I implemented a combined heat and power plant - not planning
on introducing this into energy_py v2.0.

I plan to make energy_py environments fully agent agnostic by following the Open AI Gym schema.

**Agents**

The following agents are currently implemented:

- [Random agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/random_agent.py)

- [Naive battery agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/naive/naive_battery.py)

- [DQN aka Q-Learning with experience replay and target network](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/Q_learning/dqn.py)

- [Deterministic Policy Gradient](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/Q_learning/dpg.py)
