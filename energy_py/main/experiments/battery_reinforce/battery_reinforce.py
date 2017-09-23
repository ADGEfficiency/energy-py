"""
This experiment script uses a REINFORCE agent to control the battery environment.

Experiment runs through the entire length of the state time series CSV.
"""

import sys

import argparse
import tensorflow as tf

from energy_py.agents.reinforce.agent import REINFORCE_Agent
from energy_py.envs.battery.battery_env import Battery_Env
from energy_py.main.scripts.experiment_blocks import run_single_episode
from energy_py.main.scripts.visualizers import Eternity_Visualizer

#  can probably make this into an episode block
parser = argparse.ArgumentParser(description='battery REINFORCE experiment')
parser.add_argument('--episodes', type=int, default=10,
                    help='number of episodes to run (default: 10)')
parser.add_argument('--episode_length', type=int, default=48,
                    help='length of a single episode (default: 48)')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='agent optimizer learning rate (default: 0.01)')
args = parser.parse_args()

EPISODES = args.episodes
EPISODE_LENGTH = args.episode_length
LEARNING_RATE = args.learning_rate

print('running {} episodes of length {}'.format(EPISODES, EPISODE_LENGTH))

env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  episode_start  = 10000,
                  power_rating   = 2,  #  in MW
                  capacity       = 4,  #  in MWh
                  initial_charge = 50,  #  in % of capacity
                  verbose        = 0)

"""
can probably pull the 4/5 out as the true input
for the episode decay input
"""

agent = REINFORCE_Agent(env,
                        epsilon_decay_steps = EPISODE_LENGTH * EPISODES * 4 / 5,
                        learning_rate = LEARNING_RATE,
                        batch_size = 64)

#  creating the TensorFlow session for this experiment
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(1, EPISODES):
        agent, env, sess = run_single_episode(episode,
                                               agent,
                                               env,
                                               sess)

        #  get a batch to learn from
        observations, actions, returns = agent.memory.get_episode_batch(episode)
        #  train the model
        loss = agent.learn(observations, actions, returns, sess)

    #  now run one zero exploration episode
    agent.epsilon_greedy.mode = 'testing'
    agent, env, sess = run_single_episode(episode,
                                           agent,
                                           env,
                                           sess)

#  finally collect data from the agent & environment
global_history = Eternity_Visualizer(episode, agent, env)
outputs = global_history.output_results()
