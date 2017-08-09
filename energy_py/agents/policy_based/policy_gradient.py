"""
https://github.com/yukezhu/tensorflow-reinforce/blob/master/run_reinforce_cartpole.py
https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/pg_reinforce.py

can do a gaussian policy:
https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb

References
[1] = https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
notes
    use of tf.multinominal to sample across discrete action space
"""

import tensorflow as tf

from core_agent import Agent_Memory

def fc_layer(input_tensor, wt_shape, bias_shape, activation=None):
    wt_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(0)

    W = tf.get_variable('W', wt_shape, initializer=wt_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)
    matmul = tf.matmul(input_tensor, W) + b

    if activation:
        output = activation(matmul)
    else:
        output = matmul
    return output
