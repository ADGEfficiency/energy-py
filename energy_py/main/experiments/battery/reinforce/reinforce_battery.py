"""
This experiment script uses a REINFORCE agent to control the battery environment.

Experiment runs through the entire length of the state time series CSV.
"""

import sys

import tensorflow as tf

from energy_py.agents.policy_based.reinforce import REINFORCE_Agent
from energy_py.envs.battery.battery_env import Battery_Env
from energy_py.main.scripts.experiment_blocks import run_single_episode
from energy_py.main.scripts.visualizers import Eternity_Visualizer
args = sys.argv

EPISODES = int(args[1])
EPISODE_LENGTH = int(args[2])

print('running {} episodes of length {}'.format(EPISODES, EPISODE_LENGTH))

env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  episode_start  = 0,
                  power_rating   = 2,  #  in MW
                  capacity       = 4,  #  in MWh
                  verbose        = 1)
print('made env')
agent = REINFORCE_Agent(env,
                        epsilon_decay_steps = EPISODE_LENGTH * EPISODES / 2,
                        learning_rate = 0.1,
                        batch_size = 64)
print('made agent')
#  creating the TensorFlow session for this experiment
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(1, EPISODES):
        agent, env, sess = run_single_episode(episode,
                                               agent,
                                               env,
                                               sess)

        #  now we start the learning process_episode
        #  this should be a function TODO
        #  get a batch to learn from
        observations, actions, returns = agent.memory.get_batch(agent.batch_size)
        #  train the model
        loss = agent.learn(observations, actions, returns, sess)

#  finally collect data from the agent & environment
global_history = Eternity_Visualizer(episode, agent, env)
outputs = global_history.output_results()
