## energy_py v2.0

**energy_py is reinforcement learning for energy systems.** It's a collection of agents and environments built in Python.

The goal is to demonstrate that reinforcement learning agents can control virtual energy environments.  Proving this virtually is the first step towards using reinforcment learning in real world energy systems.

energy_py supports this goal by providing a collection of reinforcement learning agents, energy environments and tools to run experiments.  

This project is built and maintained by Adam Green - adam.green@adgefficiency.com.  I write about energy & machine learning at [adgefficiency.com](http://adgefficiency.com/).

### Basic usage
```
from energy_py.agents import DQN, Keras_ActionValueFunction
from energy_py.envs import Battery_Env

env = Battery_Env(lag            = 0,
                  episode_length = 2016,
                  episode_start  = 0,
                  power_rating   = 2,  #  in MW
                  capacity       = 2,  #  in MWh
                  initial_charge = 0,  #  in % of capacity
                  round_trip_eff = 1.0, #  in % - 80-90% in practice
                  verbose        = False)

agent = DQN(env,
            Q_actor=Keras_ActionValueFunction,
            Q_target=Keras_ActionValueFunction,
            discount=0.9)
```

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
The main dependencies of energy_py are numpy, pandas, Keras & TensorFlow (GPU).  

energy_py was built using Keras 2.0.8 & TensorFlow 1.3.0.  

### Project structure

All classes inherit from the [Utils](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/scripts/utils.py) class, which contains useful generic functionality.

Environments are created by inheriting from the [Base_Env](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/env_core.py) class.  Agents are created by inheriting from the [Base_Agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/agent_core.py) class.  Another key object is the [Agent Memory](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/memory.py) which holds and process agent experience.  

**Environments**

The main focus is on building the [battery environment](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/battery).  This is for two reasons - first a battery is site agnostic so the environment can be built without site knowledge.  The second is that electric batteries are key to enabling intermittent renewables to compete with dispatchable renewables.

I am also working on a [cooling flexibility action](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/precool) and have previously implemented a combined heat and power plant as a reinforcement learning environment in energy_py v1.0.

**Agents**

The following agents are currently implemented:

- [Naive battery agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/naive/naive_battery.py)

- [REINFORCE aka Monte Carlo policy gradient - TensorFlow](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/policy_based/reinforce.py)

- [DQN aka Q-Learning with experience replay and target network - Keras](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/Q_learning/DQN.py)

I plan to make energy_py environments fully agent agnostic - so that agents built using other libraries can be used.

**Function approximators**

energy_py is deep learning library agnostic - any framework can be used to [parameterize either policies or value functions](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/agents/function_approximators).  

### Experiments

It's envionsed that energy_py will be used to run experiments.  Currently three are implemented:

- [Naive agent + battery environment experiment](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/naive/naive_battery.py)

- [REINFORCE agent + battery environment experiment](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/reinforce/reinforce_battery.py)

- [DQN + battery environment experiment](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/DQN_battery.py)
