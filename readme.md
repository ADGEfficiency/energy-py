**energy_py is reinforcement learning for energy systems.** It is a collection of reinforcement learning agents and environments built in Python.

This aim of this project is to demonstrate the ability of reinforcement learning to control virtual energy environments.  Using reinforcement learning to control energy systems requires first proving the concepts in a virtual environment.

This project is built and maintained by Adam Green - adam.green@adgefficiency.com.

### Installation
conda create --name energy_py python=3.5.2

cd energy_py
python setup.py install

pip install requirements.txt

### Guide to the project

**Environments**
Electric battery charging & discharging
Cooling flexibility action

**Agents**
REINFORCE aka Monte Carlo policy gradient - TensorFlow


**Basic usage**
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
