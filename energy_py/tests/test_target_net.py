"""
Test that the target network update is working correctly
"""

import tensorflow as tf
from energy_py.envs import CartPoleEnv
from energy_py.agents import DQN


agent_config = {'discount': 0.97,
                'total_steps': 10,
                'batch_size': 32,
                'layers': (50, 50),
                'learning_rate': 0.0001,
                'initial_random': 0.0,
                'epsilon_decay_fraction': 0.3,
                'memory_fraction': 0.4,
                'memory_type': 'priority',
                'process_observation': False,
                'process_target': False}


with tf.Session() as sess:
    env = CartPoleEnv()

    agent_config['env'] = env
    agent_config['sess'] = sess
    agent_config['tau'] = 0.5

    agent = DQN(**agent_config)

    sess.run(tf.global_variables_initializer())

    for online, target in zip(agent.online.params, agent.target.params):
        o, t = sess.run(online), sess.run(target)

        assert o.all() == t.all()

    
