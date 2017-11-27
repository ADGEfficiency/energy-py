"""
This experiment script uses the Monte Carlo REINFORCE agent
to control the battery environment.
"""

import logging
import sys

import argparse
import tensorflow as tf

from energy_py import run_single_episode, EternityVisualizer, Utils
from energy_py.agents import REINFORCE, GaussianPolicy
from energy_py.envs import BatteryEnv

if __name__ == '__main__':
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
    parser.add_argument('--v', type=int, default=1,
                        help='controls printing (default: 1')

    args = parser.parse_args()

    EPISODES = args.ep
    EPISODE_LENGTH = args.len
    LEARNING_RATE = args.lr
    DISCOUNT = args.gamma
    OUTPUT_RESULTS = args.out
    VERBOSE = args.v

    RESULTS_PATH = 'reinforce_results/'
    ARGS_PATH = RESULTS_PATH + 'args.txt'
    LOG_PATH = RESULTS_PATH + 'logs.log'
    BRAIN_PATH = RESULTS_PATH + 'brain/'
    utils = Utils().save_args(args, path=ARGS_PATH)

    #  using a single root logger for all modules
    #  can do better but just happy to have this working at the moment!
    logging.basicConfig(level=logging.INFO)

    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(LOG_PATH)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)
                        

#  first we create our environment
    env = BatteryEnv(episode_length=EPISODE_LENGTH,
                     initial_charge=0.0,  #  in % of capacity
                     round_trip_eff=1.0)  #  in % - 80-90% in practice

    logging.info('created {}MW battery {}MWh of storage'.format(env.power_rating,
                                                               env.capacity))

#  now we create our agent with a Gaussian policy
    agent = REINFORCE(env,
                      DISCOUNT,
                      BRAIN_PATH,
                      policy=GaussianPolicy,
                      lr=LEARNING_RATE,
                      process_reward=None,
                      process_return=None)

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
            observations, actions, rewards = agent.memory.get_episode_batch(episode,
                                                                            scaled_actions=False)
            returns = agent.calculate_returns(rewards)

            #  train the model
            loss = agent.learn(observations=observations,
                               actions=actions,
                               discounted_returns=returns,
                               session=sess)

            if episode % OUTPUT_RESULTS == 0:
                #  collect data from the agent & environment
                global_history = Eternity_Visualizer(episode, agent, env,
                                                     results_path=RESULTS_PATH)
                outputs = global_history.output_results(save_data=False)
