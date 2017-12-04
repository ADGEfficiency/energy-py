import numpy as np
import tensorflow as tf

from energy_py.agents import tfValueFunction

model_dict = {'input_nodes': 4,
              'output_nodes': 6,
              'layers': [3],
              'lr': 0.1}

def test_train_op():
    v = tfValueFunction(model_dict)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    before = sess.run(tf.trainable_variables())

    _ = sess.run(v.train_op,
                 feed_dict={v.obs: np.arange(4).reshape(-1,4),
                            v.target: np.arange(6).reshape(-1,6)})
    after = sess.run(tf.trainable_variables())

    for b, a in zip(before, after):
        assert(b != a).any()


def test_target_network():

    Q_actor = tfValueFunction(model_dict, 'actor')
    Q_target = tfValueFunction(model_dict, 'target')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    act_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

    before_actor = sess.run(act_vars)
    before_target = sess.run(target_vars)

    #  train only the actor network
    _ = sess.run(Q_actor.train_op,
                 feed_dict={Q_actor.obs: np.arange(4).reshape(-1,4),
                            Q_actor.target: np.arange(6).reshape(-1,6)})

    after_actor = sess.run(act_vars)
    after_target = sess.run(target_vars)

    #  make sure that actor vars did change
    for b, a in zip(before_actor, after_actor):
        assert (a != b).any()
    
    #  make sure target vars didn't change
    for b, a in zip(before_target, after_target):
        assert (a == b).all() 
