"""
This experiment script uses a naive agent to control the battery environment.

Experiment runs through the entire length of the state time series CSV.
"""

from energy_py.agents.naive.battery import Naive_Battery_Agent
from energy_py.envs.battery.env import Battery_Env

EPISODE_LENGTH = 7931  #  entire length of the CSV

env = Battery_Env(lag            = 0,      #  observation = state
                  episode_length = EPISODE_LENGTH,
                  power_rating   = 2,      #  in MW
                  capacity       = 2,
                  verbose        = 1)     #  in MWh

agent = Naive_Battery_Agent(env=env)

done, step = False, 0

observation = env.get_observation(step, env.initial_charge)
while done is False:
    #  select an action
    action = agent.act(observation)
    #  take one step through the environment
    next_observation, reward, done, info = env.step(action)
    #  store the experience
    agent.memory.add_experience((observation, action, reward, next_observation, step))
    step += 1
    observation = next_observation
