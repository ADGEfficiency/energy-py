## energy_py v2.0

**energy_py is reinforcement learning for energy systems.**

Proving that reinforcement learning agents can control virtual energy environments is the first step towards using reinforcement learning to optimize real world energy systems.

energy_py supports this goal by providing a **collection of reinforcement learning agents, energy environments and tools to run experiments.**

This project is built and maintained by Adam Green - [adam.green@adgefficiency.com](adam.green@adgefficiency.com).  

I write about energy & machine learning at [adgefficiency.com](http://adgefficiency.com/).  

I teach a one day [introduction to reinforcement learning learning](https://github.com/ADGEfficiency/DSR_RL) class at [Data Science Retreat](https://www.datascienceretreat.com/).

### Basic usage
```
from energy_py.agents import DQN, KerasQ
from energy_py.envs import BatteryEnv

env = BatteryEnv()

agent = DQN(env,
            discount=0.9,
            Q=KerasQ,           # Keras model to approximate Q(s,a)
            batch_size=64,
            brain_path='/brain')

obs = env.reset()
action = agent.act(observation=obs)
next_obs, reward, done, info = env.step(action)
agent.memory.add_experience(obs, action, reward, next_obs, step, episode)

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

Environments are created by inheriting from the [BaseEnv](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/envs/env_core.py) class.  Agents are created by inheriting from the [BaseAgent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/agent_core.py) class.  Another key object is the [AgentMemory](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/memory.py) which holds and processes agent experience.  

**Environments**

Currently the main focus is on building the [battery environment](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/battery).  This is for two reasons - first a battery is site agnostic so the environment can be built without site knowledge.  The second is that electric batteries will be key to increasing the penetration of intermittent renewables.

I am also working on a [cooling flexibility action](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/envs/precool) and have previously implemented a Combined Heat and Power plant as a reinforcement learning environment in energy_py v1.0.

**Agents**

The following agents are currently implemented:

- [Naive battery agent](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/naive/naive_battery.py)

- [REINFORCE aka Monte Carlo policy gradient - TensorFlow](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/policy_based/reinforce.py)

- [DQN aka Q-Learning with experience replay and target network - Keras](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/agents/Q_learning/DQN.py)

I plan to make energy_py environments fully agent agnostic - so that agents built using other libraries can be used.

**Function approximators**

energy_py is deep learning library agnostic - any framework can be used to [parameterize either policies or value functions](https://github.com/ADGEfficiency/energy_py/tree/master/energy_py/agents/function_approximators).  Classes are used to allow flexibility in combining different function approximator with different agents.

### Experiments

energy_py can be used to run experiments.  Currently three are implemented:

- [Naive agent + battery](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/naive/naive_battery.py)

- [REINFORCE agent + battery](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/reinforce/reinforce_battery.py)

- [DQN + battery](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/experiments/battery/DQN_battery.py)
