"""
This experiment script uses a REINFORCE agent to control the battery environment.

Experiment runs through the entire length of the state time series CSV.
"""

import tensorflow as tf

from energy_py.agents.policy_based.reinforce import REINFORCE_Agent
from energy_py.envs.battery.battery_env import Battery_Env
from energy_py.main.scripts.experiment_blocks import run_single_episode

EPISODE_LENGTH = 2016
EPISODES = 20

env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  power_rating   = 2,      #  in MW
                  capacity       = 2,
                  verbose        = 0)     #  in MWh

agent = REINFORCE_Agent(env,
                        learning_rate = 0.001,
                        batch_size = 64,
                        epsilon_decay_steps = EPISODE_LENGTH * EPISODES / 2)

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

    #  finally output resutls from the environment and agent
    env_outputs = agent.env.output_results()
    agent_outputs = agent.memory.output_results()
