import time

import numpy as np
import pandas as pd

import agents.Double_Q_Learner
import environments.env_CHP
import outputs

EPISODES = 10  # naive episode inclusive
EPISODE_LENGTH = 336  # number of half hour periods per episode
LAG = 1  # positive = see future.  negative = can't see present
RANDOM_TS = True
GEN_OUTPUTS = 100  # generate outputs every n episodes
RUN_NAME = 'EPISODE_LENGTH_'+str(EPISODE_LENGTH)

env = environments.env_CHP.env(episode_length=EPISODE_LENGTH,
                               lag=LAG,
                               random_ts=RANDOM_TS,
                               verbose=1)
print(env.verbose)

agent = agents.Double_Q_Learner.agent(env,
                         verbose=1,
                         device=0,
                         run_name=RUN_NAME)

print('Started at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('Running naive episode')
episode, agent.policy_ = 1, 0  # naive policy
agent.single_episode(episode)

e_greedy_episodes = np.arange(2, EPISODES+1)
for episode in e_greedy_episodes:
    agent.policy_ = 1  # e-greedy policy
    agent.single_episode(episode)
    if episode != 1 and episode % GEN_OUTPUTS == 0:
        summary = agent.create_outputs()

print('Running greedy episode')
episode += 1
agent.epsilon = 0  # greedy
agent.single_episode(episode)
agent.save_csv = True
summary = agent.create_outputs()
