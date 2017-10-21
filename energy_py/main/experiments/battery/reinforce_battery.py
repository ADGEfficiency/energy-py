"""
This experiment script uses a REINFORCE agent to control the battery environment.

Experiment runs through the entire length of the state time series CSV.
"""

import sys

import argparse
import tensorflow as tf

from energy_py.agents.reinforce.monte_carlo import MC_Reinforce
from energy_py.agents.function_approximators.tensorflow import GaussianPolicy

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

OUTPUT_RESULTS = 50  #  output results every n episodes

print('running {} episodes of length {}'.format(EPISODES, EPISODE_LENGTH))
print('learning rate is {}'.format(LEARNING_RATE))

#  first we create our environment
env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  episode_start  = 10000,
                  power_rating   = 2,  #  in MW
                  capacity       = 4,  #  in MWh
                  initial_charge = 0,  #  in % of capacity
                  verbose        = False)

#  now we create our agent with a Gaussian policy
agent = MC_Reinforce(env,
                        GaussianPolicy,
                        model_dict={'layers' : [100, 100, 100]},
                        learning_rate = LEARNING_RATE,
                        discount = 0.99,
                        verbose=True)

#  creating the TensorFlow session for this experiment
with tf.Session() as sess:
    #  initalizing TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for episode in range(1, EPISODES):
        agent, env, sess = run_single_episode(episode,
                                               agent,
                                               env,
                                               sess)

        #  get a batch to learn from
        #  note that we don't scale actions as we need to take logprob(action)
        observations, actions, returns = agent.memory.get_episode_batch(episode,
                                                                        scaled_actions=False)
        #  train the model
        loss = agent.learn(observations=observations,
                           actions=actions,
                           discounted_returns=returns,
                           session=sess)


        if episode % OUTPUT_RESULTS == 0:
            #  collect data from the agent & environment
            global_history = Eternity_Visualizer(episode, agent, env,
                                                 results_path='reinforce_results/')
            outputs = global_history.output_results(save_data=False)
