## energy_py v2.0

**energy_py is reinforcement learning for energy systems.** It is a collection of reinforcement learning agents and environments built in Python.

This aim of this project is to demonstrate the ability of reinforcement learning to control virtual energy environments.  Using reinforcement learning to control energy systems requires first proving the concepts in a virtual environment.

The goal of the project is to provide a collection of reinforcement learning agents, energy environments and tools to run experiments.  

**v2.0 was a complete rework of the entire project.**  v1.0 had agents & environments that are not yet ported into v2.0.  

This project is built and maintained by Adam Green - adam.green@adgefficiency.com.  I write about energy & machine learning at [adgefficiency.com](http://adgefficiency.com/).
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

energy_py was built using TensorFlow 1.3.0.

### Vision for energy_py
The value of energy_py will be in the environment models.

There are many reinforcement learning libraries that offer higher quality implementations of agents.  It's planned that energy_py will hold a basic set of agents - but that most of the work will be done using agents from other packages.  

### Project structure

All classes inherit from the Utils class, which contains useful generic functionality.

Environments are created by inheriting from the [Base_Env](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/env_core.py) class.  

Agents are created by inheriting from the [Base_Agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/agent_core.py) class.  

**Environments**

The current focus is on building the [battery environment](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/battery).  THis is for two reasons - first a battery is site agnostic so the environment can be built without site knowledge.  The second is that electric batteries are key to enabling intermittent renewables to compete with dispatchable renewables.

In the future I plan to add more environments that allow agents to optimize clean energy systems:
- [A cooling flexibility action (in progress)](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/precool)
- Combined heat and power plants (to be ported over from energy_py v1.0)

**Agents**

[Naive battery agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/naive/naive_battery.py)

[REINFORCE aka Monte Carlo policy gradient - TensorFlow](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/policy_based/reinforce.py)

Q-Learning (to be ported over from energy_py v1.0)

Double Q-Learning (to be ported over from energy_py v1.0)

### Experiments

It's envionsed that energy_py will be used to run experiments.  Currently two are implemented.

[Naive agent + battery environment experiment](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/naive/naive_battery.py)
```
cd energy_py/energy_py/main/experiments/experiments/battery/naive/naive_battery.py

python naive_battery.py
```
[reinforce agent + battery environment experiment](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/reinforce/reinforce_battery.py)

Note that this experiment has two arguments - number of episodes & episode length.
```
cd energy_py/energy_py/main/experiments/experiments/battery/naive/naive_battery.py

python naive_battery.py 10 3000
```

### Basic usage
```
from energy_py.agents.policy_based.reinforce import REINFORCE_Agent
from energy_py.envs.battery.battery_env import Battery_Env
from energy_py.main.scripts.experiment_blocks import run_single_episode

env = Battery_Env()
agent = REINFORCE_Agent(env)

with tf.Session() as sess:
     agent, env, sess = run_single_episode(episode,
                                               agent,
                                               env,
                                               sess)
```
