"""
This experiment script uses the DQN agent
to control the battery environment.
"""
import sys

import argparse

from energy_py.agents import DQN, Keras_ActionValueFunction

from energy_py.envs import Battery_Env
from energy_py import run_single_episode
from energy_py.scripts.visualizers import Eternity_Visualizer
from energy_py import Utils

#  use argparse to collect command line arguments
parser = argparse.ArgumentParser(description='battery REINFORCE experiment')
parser.add_argument('--ep', type=int, default=10,
                    help='number of episodes to run (default: 10)')
parser.add_argument('--len', type=int, default=48,
                    help='length of a single episode (default: 48)')
parser.add_argument('--bs', type=int, default=32,
                    help='batch size (default: 32)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount rate (default: 0.9)')
parser.add_argument('--out', type=int, default=50,
                    help='output results every n episodes (default: 50')
parser.add_argument('--v', type=int, default=0,
                    help='controls printing (default: 0')
args = parser.parse_args()

#  pull out the command line args
EPISODES = args.ep
EPISODE_LENGTH = args.len
BATCH_SIZE = args.bs
DISCOUNT = args.gamma
OUTPUT_RESULTS = args.out
VERBOSE = args.v


#  first we create our environment
env = Battery_Env(lag            = 0,
                  episode_length = EPISODE_LENGTH,
                  episode_start  = 0,
                  power_rating   = 2,  #  in MW
                  capacity       = 2,  #  in MWh
                  initial_charge = 0,  #  in % of capacity
                  round_trip_eff = 1.0, #  in % - 80-90% in practice
                  verbose        = False)

#  set up some hyperparameters using ratios from DeepMind 2015 Atari
total_steps = EPISODES * env.state_ts.shape[0]
epsilon_decay_steps = total_steps / 2
update_target_net = max(10, int(total_steps / (env.state_ts.shape[0] * 100)))
memory_length = int(total_steps/10)

#  save the hyperparameters
utils = Utils()
_ = utils.save_args(args,
                    path='DQN_results/args.txt',
                    optional={'total steps':total_steps,
                              'epsilon decay steps':epsilon_decay_steps,
                              'update_target_net':update_target_net,
                              'memory length':memory_length})

#  now we create our agent
agent = DQN(env=env,
            Q=Keras_ActionValueFunction,
            discount=DISCOUNT,
            batch_size=BATCH_SIZE,
            epsilon_decay_steps=epsilon_decay_steps,
            epsilon_start=1.0,
            update_target_net=update_target_net,
            memory_length=memory_length,
            scale_targets=True,
            brain_path='DQN_results/brain',
            load_agent_brain=False,
            verbose=VERBOSE)

for episode in range(1, EPISODES):

    #  initialize before starting episode
    done, step = False, 0
    observation = env.reset(episode)

    #  while loop runs through a single episode
    while done is False:
        #  select an action
        action = agent.act(observation=observation)
        #  take one step through the environment
        next_observation, reward, done, info = env.step(action)
        #  store the experience
        agent.memory.add_experience(observation, action, reward, next_observation, step, episode)
        step += 1
        observation = next_observation

        #  get a batch to learn from
        obs, actions, rewards, next_obs = agent.memory.get_random_batch(BATCH_SIZE)

        #  train the model
        #  can't train before memory > batch_size
        #  usually spend a few episodes without learning
        #  to smooth distribution of data in memory (a bit!)
        if episode >= 2:
            loss = agent.learn(observations=obs,
                               actions=actions,
                               rewards=rewards,
                               next_observations=next_obs,
                               episode=episode)

    if episode % update_target_net == 0:
        agent.update_target_network()

    if episode % OUTPUT_RESULTS == 0:
        #  collect data from the agent & environment
        global_history = Eternity_Visualizer(episode, agent, env,
                                             results_path='DQN_results/')
        outputs = global_history.output_results(save_data=False)

        agent.save_brain()
