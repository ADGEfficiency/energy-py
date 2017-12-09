import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DeterminsticPolicy(object):
    """
    A policy that maps state to action 
    For use in a DeterminsticPolicyGradient agent

    args
       model_dict (dict) 
    """
    def __init__(self, model_dict, scope='DeterminsticPolicy'):
        
        #  assert action space symmetric 
        assert (action_space.high == -action_space.low)

        #  actor network
        self.obs, self.actor_net = self.make_acting_graph()
        self.network_param = tf.trainable_variables()

        #  target network
        self.target_net = self.make_acting_graph()
        self.target_param = tf.trainable_variables()[len(self.network_params):]
        self.num_train_vars = len(self.network_param) + len(self.target_param)
        assert len(tf.trainable_variables()) == self.num_train_vars

        self.train_op = self.make_learning_graph()

    def update_target_net(self):
        """
        Operation for copying actor weights to target
        """
        target_params = []

        for param, target in zip(self.network_param, self.target_param):
            target.assign(tf.multiply(param, self.tau) +
                          tf.multiply(target, 1 - self.tau))

    def make_acting_graph(self):
        """
        Creates the TensorFlow graph for selecting actions determinstically.

        """
        obs = tf.placeholder(tf.float32,
                                  [None, self.input_nodes],
                                  'observation')
        #  add the input layer
        with tf.variable_scope('input_layer'):
            layer = tf.layers.dense(inputs=self.obs,
                                    units=self.layers[0])

            #  set training=True so that each batch is normalized with 
            #  statistics of the current batch
            batch_norm = tf.layers.batch_normalization(layer,
                                                       training=True)
            relu = tf.relu(batch_norm)

        #  iterate over self.layers
        for i, nodes in enumerate(self.layers[1:]):
            with tf.variable_scope('input_layer_{}'.format(i)):
                layer = tf.layers.dense(inputs=relu,
                                        units=nodes)
            batch_norm = tf.layers.batch_normalization(layer,
                                                       training=True)
            relu = tf.relu(batch_norm)

        #  use tanh activation function to squeeze output to -1, +1 
        with tf.variable_scope('output_layer'):
            wt_init = tf.random_uniform_initializer(minval=-0.003,
                                                    maxval=0.003)
            out = tf.layers.dense(inputs=layer,
                                  units=self.output_nodes,
                                  activation='tanh',
                                  kernel_initializer=wt_init)

            actions = tf.multiply(out, self.action_space.high)

        return obs, actions

    def make_learning_graph(self):
        """

        """
        #  this gradient is provided by the critic
        self.action_gradient = tf.placeholder(tf.float32, [None,
                                                           self.action_dim])

        #  combine the gradients
        actor_gradients = tf.gradients(self.prediction,
                                       self.network_param,
                                       -self.action_gradient)
        #  clip using global norm
        self.actor_gradients = tf.clip_by_global_norm(actor_gradients)

        #  define the optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = self.optimizer.apply_gradients(zip(self.actor_gradients,
                                                      self.network_param))
        return train_op 

    def get_action(self, sess, observation):
        assert observation.shape[0] == 1
            
        #  generating an action from the policy network
        action = session.run([self.action], {self.obs : observation})

        #  create a dictionary to send action back
        output = {'action': action}

        return output['action'].reshape(int(self.output_nodes/2)), output

    def improve(self, 
                session,
                observations,
                act_grads):

        assert observations.shape[0] == act_grads.shape[0]

        feed_dict = {self.obs : observations,
                     self.action_gradient: act_grads}

        loss = session.run([self.train_op], feed_dict)

class DPGCritic(object):
    """
    An on-policy estimate of Q(s,a) for the actor policy

    Single output node for Q(s,a) of the action taken by the policy
    """
    def __init__(self, model_dict, scope='DPGCritic'):

        assert len(self.layers) >= 2
        self.obs, self.actions, self.pred = self.make_pred_graph()

        self.train_op, self.loss, self.action_grads = self.make_learning_graph()

    def make_pred_graph(self):

        obs = tf.placeholder(tf.float32,
                             [None, self.obs_dim],
                             'observation')

        act = tf.placeholder(tf.float32,
                             [None, self.a_dim],
                             'action')

        with tf.variable_scope('input_layer'):
            net = tf.layers.dense(inputs=obs,
                                    units=self.layers[0])

            #  set training=True so that each batch is normalized with 
            #  statistics of the current batch
            net = tf.layers.batch_normalization(net, training=True)
            net = tf.relu(batch_norm)

        #  add the actions into the second layer
        with tf.variable_scope('hidden_layer_1'):
            l1 = tf.layers.dense(inputs=net, self.layers[1])
            l2 = tf.layers.dense(inputs=act, self.layers[1])

            net = tf.matmul(net, l1.W) + tf.matmul(act, l2.W) + l2.b
            net = tf.layers.batch_normalization(net, training=True)

        #  iterate over self.layers
        for i, nodes in enumerate(self.layers[2:]):
            with tf.variable_scope('hidden_layer_{}'.format(i)):
                net = tf.layers.dense(inputs=net, units=nodes)
                net = tf.layers.batch_normalization(net, training=True)
                net = tf.relu(net)

        #  use tanh activation function to squeeze output to -1, +1 
        with tf.variable_scope('output_layer'):
            wt_init = tf.random_uniform_initializer(minval=-0.003,
                                                    maxval=0.003)
            out = tf.layers.dense(inputs=net,
                                  activation='linear',
                                  kernel_initializer=wt_init)
        return obs, act, out

    def make_learning_graph(self):
        #  a Bellman target created externally to this value function
        target = tf.placeholder(tf.float32,
                                [None, 1],
                                'Bellman_target')

        #  add the optimizer, loss function & train operation
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        loss = tf.losses.huber_loss(target, self.out)
        train_op = self.optimizer.minimize(loss)

        #  gradient of network parameters wrt action 
        action_grads = tf.gradients(self.out, self.action)
        return target, train_op, action_grads
