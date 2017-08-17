"""
This experiment script uses a naive agent to control the battery environment.

Experiment runs through the entire length of the state time series CSV.
"""

from energy_py.agents.naive.naive_battery import Naive_Battery_Agent
from energy_py.envs.battery.battery_env import Battery_Env
from energy_py.main.scripts.experiment_blocks import run_single_episode

EPISODE_LENGTH = 2016
EPISODES = 5

env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  power_rating   = 2,      #  in MW
                  capacity       = 2 * 5,  #  in MWh
                  verbose        = 0)

assert env.lag == 0  # must be zero for naive agents

agent = Naive_Battery_Agent(env=env)

for episode in range(1, EPISODES):
    agent, env, _ = run_single_episode(episode,
                                       agent,
                                       env)

#  episodes are now all finished
#  output resutls from the environment and agent
env_outputs = agent.env.output_results()
agent_outputs = agent.memory.output_results()

env_vis = agent.env.episode_visualizer
