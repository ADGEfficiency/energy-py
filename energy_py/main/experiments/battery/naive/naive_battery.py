"""
This experiment script uses a naive agent to control the battery environment.

This experiment can be run for the entire length of the episode by
    EPISODE_LENGTH = 'maximum'
    EPISODE_START  = 0

RL agents can benefit from training from random periods of the time series.

The naive agent needs to be run only once - because both the policy and the
environment are determinstic.

Code below runs a single episode & saves results.
"""

from energy_py.agents.naive.naive_battery import Naive_Battery_Agent
from energy_py.envs.battery.battery_env import Battery_Env
from energy_py.main.scripts.experiment_blocks import run_single_episode
from energy_py.main.scripts.visualizers import Eternity_Visualizer

EPISODE_LENGTH = 100
EPISODE_START = 0

env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  episode_start  = EPISODE_START,
                  power_rating   = 2,  #  in MW
                  capacity       = 4,  #  in MWh
                  verbose        = 1)

assert env.lag == 0  # must be zero for naive agents

agent = Naive_Battery_Agent(env=env)
episode = 1
agent, env, _ = run_single_episode(episode,
                                   agent,
                                   env)

outputs = Eternity_Visualizer(episode, agent, env).output_results()
