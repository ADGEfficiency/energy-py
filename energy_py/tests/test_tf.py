"""
"""

import numpy as np
import tensorflow as tf

from energy_py.agents import Qfunc

epsilon = 1e-5

config = {'input_shape': (2,),
          'layers': (4, 8),
          'output_shape':(8,),
          'learning_rate': 0.1}

input_shape = config['input_shape']
layers = config['layers']
output_shape = config['output_shape']

def weight(shape): return np.ones(shape)
def bias(shape): return np.ones(shape)
def relu(x): return x * (x > 0)

observation = np.random.uniform(size=input_shape).reshape(1, -1)
action = np.array([0, 1]).reshape(1, 2)
target = np.ones((1,1))
weights = np.ones(1).reshape(1,1)

def test_train_op():
    with tf.Session() as sess:
        q = Qfunc(config, scope='test_train_op')

        sess.run(tf.global_variables_initializer())

        params = [p for p in tf.trainable_variables()
                  if p.name.startswith('test_train_op')]

        before = sess.run(params)

        _ = sess.run(q.train_op,
                     feed_dict={q.observation: observation,
                                q.target: target,
                                q.action: action,
                                q.importance_weights: weights}) 

        after = sess.run(params)

        for b, a in zip(before, after):
            assert(b != a).any()

from energy_py.envs import CartPoleEnv
from energy_py.agents import DQN

def test_target_net():
    agent_config = {'discount': 0.97,
                    'total_steps': 100,
                    'batch_size': 2,
                    'layers': (50, 50),
                    'learning_rate': 0.0001,
                    'initial_random': 0.0,
                    'epsilon_decay_fraction': 0.3,
                    'memory_fraction': 0.1,
                    'memory_type': 'deque',
                    'process_observation': False,
                    'process_target': False}


    with tf.Session() as sess:
        env = CartPoleEnv()

        agent_config['env'] = env
        agent_config['sess'] = sess
        agent_config['tau'] = 0.5

        agent = DQN(**agent_config)

        ls = []
        for online, target in zip(agent.online.params, agent.target.params):
            o, t = sess.run(online), sess.run(target)
            np.testing.assert_allclose(o, t)


