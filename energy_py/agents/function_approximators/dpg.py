import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DPGActor(object):
    """
    Policy that maps state to a single continuous action

    args
       model_dict (dict) 
    """
    def __init__(self, model_dict, scope='DPG_Actor'):
        
        self.input_nodes = model_dict['input_nodes']
        self.output_nodes = model_dict['output_nodes']
        self.layers = model_dict['layers']

        self.lr = model_dict['lr']
        self.tau = model_dict['tau']

        self.action_space = model_dict['action_space']

        #  actor network
        with tf.variable_scope('actor_online_net'):
            self.obs, self.action = self.make_acting_graph()
        self.network_param = tf.trainable_variables()

        #  target network
        with tf.variable_scope('actor_target_net'):
            self.t_obs, self.t_action = self.make_acting_graph()
        self.t_network_param = tf.trainable_variables()[len(self.network_param):]

        self.num_vars = len(self.network_param) + len(self.t_network_param)

        #  tf machinery to improve actor network 
        self.train_op = self.make_learning_graph()

        #  exploration noise
        self.actor_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.action_space.shape[0]))

    def make_acting_graph(self):
        """
        """
        obs = tf.placeholder(tf.float32, [None, self.input_nodes], 'obs')

        #  add the input layer
        with tf.variable_scope('input_layer'):
            layer = tf.layers.dense(inputs=obs,
                                    units=self.layers[0])

            #  set training=True so that each batch is normalized with 
            #  statistics of the current batch
            batch_norm = tf.layers.batch_normalization(layer,
                                                       training=True)
            relu = tf.nn.relu(batch_norm)

        #  iterate over self.layers
        for i, nodes in enumerate(self.layers[1:]):
            with tf.variable_scope('input_layer_{}'.format(i)):
                layer = tf.layers.dense(inputs=relu, units=nodes)
                batch_norm = tf.layers.batch_normalization(layer, training=True)
                relu = tf.nn.relu(batch_norm)

        with tf.variable_scope('output_layer'):
            wt_init = tf.random_uniform_initializer(minval=-0.003,
                                                    maxval=0.003)
            action = tf.layers.dense(inputs=layer,
                                     units=self.output_nodes,
                                     activation=None,
                                     kernel_initializer=wt_init)

            #  clipping the action
            lows = np.array([space.low for space in self.action_space.spaces])
            highs = np.array([space.high for space in self.action_space.spaces])
            action = tf.clip_by_value(action, lows, highs)

        return obs, action 

    def make_learning_graph(self):
        """
        Used to update the actor graph by applying gradients to 
        self.network_param
        """
        #  this gradient is provided by the critic
        self.action_gradient = tf.placeholder(tf.float32, 
                                              [None, self.action_space.shape[0]])

        #  combine the gradients
        self.actor_gradients = tf.gradients(self.action,
                                       self.network_param,
                                       -self.action_gradient)

        #  clip using global norm
        #self.actor_gradients = tf.clip_by_global_norm(actor_gradients, 5)

        #  define the optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        #  improve the actor network using the DPG algorithm
        train_op = self.optimizer.apply_gradients(zip(self.actor_gradients,
                                                      self.network_param))

        return train_op 

    def get_action(self, sess, obs):
        #  generating an action from the policy network
        determ_action = sess.run(self.action, {self.obs : obs})
        noise = self.actor_noise().reshape(-1, self.action_space.shape[0])
        action = determ_action + noise
         
        #  clipping the action
        lows = np.array([space.low for space in self.action_space.spaces])
        highs = np.array([space.high for space in self.action_space.spaces])
        action = np.clip(action, lows, highs)

        action = action.reshape(obs.shape[0],  self.action_space.shape[0])
        return determ_action, noise, action 

    def get_target_action(self, sess, obs):
        #  generates an action from the target network
        t_action = sess.run(self.t_action, {self.t_obs: obs})
        return t_action.reshape(obs.shape[0], self.action_space.shape[0])

    def improve(self, 
                sess,
                obs,
                act_grads):

        feed_dict = {self.obs: obs,
                     self.action_gradient: act_grads}

        loss = sess.run(self.train_op, feed_dict)
        return loss

    def update_target_net(self):
        """
        Operation for copying actor weights to target
        """

        for param, target in zip(self.network_param, self.t_network_param):
            target.assign(tf.multiply(param, self.tau) +
                          tf.multiply(target, 1 - self.tau))


class DPGCritic(object):
    """
    An on-policy estimate of Q(s,a) for the actor policy

    Single output node for Q(s,a) of the action taken by the policy
    """
    def __init__(self, model_dict, num_actor_vars, scope='DPG_Critic'):

        self.input_nodes = model_dict['input_nodes']
        self.output_nodes = model_dict['output_nodes']
        self.layers = model_dict['layers']

        self.lr = model_dict['lr']
        self.tau = model_dict['tau']

        self.observation_space = model_dict['observation_space']
        self.action_space = model_dict['action_space']

        with tf.variable_scope('online_net'):
            self.obs, self.action, self.pred = self.make_pred_graph()
            self.network_param = tf.trainable_variables()[num_actor_vars:]

        with tf.variable_scope('target_net'):
            self.t_obs, self.t_action, self.t_pred = self.make_pred_graph()
            self.t_network_param = tf.trainable_variables()[(len(self.network_param)
                                                                + num_actor_vars):]

        self.target, self.error, self.train_op, self.action_grads = self.make_learning_graph()

    def make_pred_graph(self):

        obs = tf.placeholder(tf.float32,
                             [None, self.observation_space.shape[0]],
                             'obs')

        act = tf.placeholder(tf.float32,
                             [None, self.action_space.shape[0]],
                             'action')

        with tf.variable_scope('input_layer'):
            net = tf.layers.dense(inputs=obs,
                                    units=self.layers[0])

            #  set training=True so that each batch is normalized with 
            #  statistics of the current batch
            net = tf.layers.batch_normalization(net, training=True)
            net = tf.nn.relu(net)

        #  add the actions into the second layer
        with tf.variable_scope('hidden_layer_1'):
            l1_W = tf.Variable(tf.random_normal([self.layers[0], self.layers[1]]))
            l2_W = tf.Variable(tf.random_normal([self.action_space.shape[0], self.layers[1]]))
            l2_b = tf.Variable(tf.zeros([self.layers[1]]))

            net = tf.matmul(net, l1_W) + tf.matmul(act, l2_W) + l2_b
            net = tf.layers.batch_normalization(net, training=True)

        #  iterate over self.layers
        for i, nodes in enumerate(self.layers[2:]):
            with tf.variable_scope('hidden_layer_{}'.format(i)):
                net = tf.layers.dense(inputs=net, units=nodes)
                net = tf.layers.batch_normalization(net, training=True)
                net = tf.nn.relu(net)

        #  use a linear layer as the output layer
        with tf.variable_scope('output_layer'):
            wt_init = tf.random_uniform_initializer(minval=-0.003,
                                                    maxval=0.003)
            out = tf.layers.dense(inputs=net,
                                  units=self.output_nodes,
                                  activation=None,
                                  kernel_initializer=wt_init)
        return obs, act, out

    def make_learning_graph(self):
        #  a Bellman target created externally to this value function
        target = tf.placeholder(tf.float32, [None, 1], 'target')

        #  add the optimizer, loss function & train operation
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        error = target - self.pred
        loss = tf.losses.huber_loss(target, self.pred)
        train_op = self.optimizer.minimize(loss)

        #  gradient of network parameters wrt action 
        action_grads = tf.gradients(self.pred, self.action)
        return target, error, train_op, action_grads

    def predict(self, sess, obs, action):

        pred = sess.run(self.pred, {self.obs: obs,
                                    self.action: action})
        return pred

    def predict_target(self, sess, obs, action):

        t_pred = sess.run(self.t_pred, {self.t_obs: obs,
                                        self.t_action: action})
        return t_pred

    def improve(self, sess, obs, action, target):
        error, loss = sess.run([self.error, self.train_op], 
                               {self.obs: obs,
                                self.action: action,
                                self.target: target})
        return error, loss

    def get_action_grads(self, sess, obs, action):
        grads = sess.run(self.action_grads, {self.obs: obs,
                                            self.action: action}) 
        return grads

    def update_target_net(self):
        """
        Operation for copying actor weights to target
        """
        for param, target in zip(self.network_param, self.t_network_param):
            target.assign(tf.multiply(param, self.tau) +
                          tf.multiply(target, 1 - self.tau))


# Taken from https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
