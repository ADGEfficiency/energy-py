"""
This experiment script uses the Monte Carlo REINFORCE agent
to control the battery environment.
"""

import sys

import argparse
import tensorflow as tf

from energy_py.agents import MC_Reinforce
from energy_py.agents.function_approximators import TF_GaussianPolicy

from energy_py.envs import Battery_Env
from energy_py import Utils
from energy_py.main.scripts.experiment_blocks import run_single_episode
from energy_py.main.scripts.visualizers import Eternity_Visualizer

#  can probably make this into an episode block
parser = argparse.ArgumentParser(description='battery REINFORCE experiment')
parser.add_argument('--ep', type=int, default=10,
                    help='number of episodes to run (default: 10)')
parser.add_argument('--len', type=int, default=48,
                    help='length of a single episode (default: 48)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='agent optimizer learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.999,
                    help='discount rate (default: 0.999)')
parser.add_argument('--out', type=int, default=50,
                    help='output results every n episodes (default: 50')
args = parser.parse_args()

EPISODES = args.ep
EPISODE_LENGTH = args.len
LEARNING_RATE = args.lr
DISCOUNT = args.gamma
OUTPUT_RESULTS = args.out

utils = Utils()
_ = utils.save_args(args,
                    path='MC_results/args.txt')

#  first we create our environment
env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  episode_start  = 0,
                  power_rating   = 2,  #  in MW
                  capacity       = 2,  #  in MWh
                  initial_charge = 0,  #  in % of capacity
                  round_trip_eff = 1.0, #  in % - 80-90% in practice
                  verbose        = False)

#  now we create our agent with a Gaussian policy
agent = MC_Reinforce(env,
                     discount=DISCOUNT,
                     policy=TF_GaussianPolicy,
                     baseline=[],
                     learning_rate=LEARNING_RATE,
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
                                                 results_path='MC_results/')
            outputs = global_history.output_results(save_data=False)
