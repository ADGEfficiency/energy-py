import numpy as np
import pandas as pd
import time
import environments
import agents
import outputs

EPISODES = 75
EPISODE_LENGTH = 336  # number of half hour periods per episode
GEN_OUTPUTS = 25  # generate outputs every n episodes

env = environments.energy_py(EPISODE_LENGTH)
agent = agents.Q_learner(env, verbose=0, device=0)
episodes = np.linspace(1, EPISODES, EPISODES, endpoint=True).astype(int)

print('Started at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('Running naive episode')
episode, agent.policy_ = 0, 0  # naive policy
agent.single_episode(episode)

for episode in episodes:
    agent.policy_ = 1  # e-greedy policy
    agent.single_episode(episode)
    if episode != 1 and episode % GEN_OUTPUTS == 0:
        memory, network_memory, summary, states, actions = outputs.gen(agent)

print('Running greedy episode')
episode += 1
agent.epsilon = 0  # greedy
agent.single_episode(episode)
agent.save_csv = True
memory, network_memory, summary, states, actions = outputs.gen(agent)
agent.network.save('results/weights.h5')

final_results = pd.DataFrame(
    data=[summary.loc[0, 'Total Reward'],
          summary.loc[episode, 'Total Reward'],
          summary.loc[episode, 'Total Reward']-summary.loc[0, 'Total Reward'],
          agent.timer.get_time(),
          episode],
    index=['Naive reward [£/episode]',
           'Optimal reward [£/episode]',
           'Value [£/episode]',
           'Run time',
           'Number of episodes'],
    columns=['Final results']
)

final_results.to_csv('results/final_results.csv')
print(final_results)
