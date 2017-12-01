"""
Purpose of this script is to test energy_py agents with gym environments.
"""
import logging

import gym
import numpy as np
import tensorflow as tf

from energy_py import EternityVisualizer
from energy_py.agents import REINFORCE, GaussianPolicy
from energy_py.envs import BatteryEnv

env = gym.envs.make('MountainCarContinuous-v0')
#env = gym.envs.make('Pendulum-v0')

DISCOUNT = 1.0

def make_paths(name):
    results = name + '/'
    paths = {'results' : results,
             'brain' : results + 'brain/',
             'logs' : results + 'logs.log',
             'args' : results + '.args.txt'}
    return paths


name = 'reinforce_gym/'
paths = make_paths(name)
BRAIN_PATH = paths['brain']
RESULTS_PATH = paths['results']
LEARNING_RATE = 0.01
EPISODES = 10000
agent = REINFORCE(env,
                                   DISCOUNT,
                                   BRAIN_PATH,
                                   policy=GaussianPolicy,
                                   lr=LEARNING_RATE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(1, EPISODES):
        state = env.reset()
        done, step = False, 0
        rewards = []
        while not done:
            env.render()    
            action = agent.act(observation=state,
                               session=sess)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            agent.memory.add_experience(state, action, reward, next_state, step, episode)

            step += 1
            state = next_state
        print('ep {} total reward {} max {}'.format(episode, 
                                                    sum(rewards),
                                                    max(rewards)))
        #  now episode is over we can learn
        
        obs, acts, rews = agent.memory.get_episode_batch(episode)

        loss = agent.learn(observations=obs,
                           actions=acts,
                           rewards=rews,
                           session=sess)

        total_rew = np.sum(rews)
        agent.memory.info['total ep rewards'].append(total_rew)

        def print_stats(total_rew):
            print('total {}'.format(np.sum(total_rew)))
            print('mean {}'.format(np.mean(total_rew)))
            print('std {} '.format(np.std(total_rew)))

        global_history = EternityVisualizer(episode, agent, env=None, results_path=RESULTS_PATH) 
        outputs = global_history.output_results(save_data=False)
