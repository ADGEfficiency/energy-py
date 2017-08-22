**energy_py is reinforcement learning for energy systems.** It is a collection of reinforcement learning agents and environments built in Python.

This aim of this project is to demonstrate the ability of reinforcement learning to control virtual energy environments.  Using reinforcement learning to control energy systems requires first proving the concepts in a virtual environment.

This project is built and maintained by Adam Green - adam.green@adgefficiency.com.

### Installation
Setup a virtual environment using your favourite method. Below I use Anaconda to create a Python 3.5 virtual environment:
```
conda create --name energy_py python=3.5.2
```
Activate the virtual environment
```
activate energy_py (windows)
or
source activate energy_py (unix)
```
Now clone the repo somewhere
```
git clone https://github.com/ADGEfficiency/energy_py.git
```
Now move into the energy_py folder and install using setup.py.  This will install energy_py into your activated Python environment
```
cd energy_py
python setup.py install
```
Finally install the required packages
```
pip install requirements.txt
```
The main packages used by energy_py are numpy, pandas & TensorFlow.  energy_py was built using TensorFlow 1.3.0.

Installing

### Guide to the project

**Environments**

[Electric battery charging & discharging](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/battery)

[A cooling flexibility action (in progress)](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/precool)

Combined heat and power plant (to be ported over from energy_py v1.0)

**Agents**

[Naive battery agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/naive/naive_battery.py)

[REINFORCE aka Monte Carlo policy gradient - TensorFlow](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/policy_based/reinforce.py)

Q-Learning (to be ported over from energy_py v1.0)

Double Q-Learning (to be ported over from energy_py v1.0)

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
