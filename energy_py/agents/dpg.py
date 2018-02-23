"""
Most functionality occurs within the Actor or Critic classes. DPG agent
only really has _act and _learn
"""

import logging

import numpy as np
import tensorflow as tf

from energy_py.agents import BaseAgent


logger = logging.getLogger(__name__)


class DPG(BaseAgent):
    """
    energy_py implementation of Determinstic Policy Gradients

    args
        env (object) energy_py environment
        actor (object) a determinstic policy
        critic (object)

    references
        Silver et. al (2014) Determinstic Policy Gradient Algorithms
        http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

    """

    def __init__(self,
                 sess,
                 env,
                 discount,
                 # actor_config,
                 # critic_config,
                 total_steps,
                 batch_size,
                 memory_fraction,
                 **kwargs):

        self.sess = sess
        self.batch_size = batch_size
        memory_length = total_steps * memory_fraction

        super().__init__(env, discount, memory_length, **kwargs)

        actor_config = {'sess': self.sess,
                        'observation_shape': self.obs_shape,
                        'action_shape': self.action_shape,
                        'action_space': self.action_space,
                        'layers': [25, 25],
                        'learning_rate': 0.0025,
                        'tau': 0.001}

        #  do I need to pass action space into critic to clip action??
        critic_config = {'sess': self.sess,
                         'observation_shape': self.obs_shape,
                         'action_shape': self.action_shape,
                         'output_nodes': 1,
                         'layers': [25, 25],
                         'learning_rate': 0.0025,
                         'tau': 0.001}

        #  create the actor & critic
        self.actor = DPGActor(**actor_config)
        self.critic = DPGCritic(**critic_config)

        self.sess.run(tf.global_variables_initializer())

    def _reset(self):
        raise NotImplementedError

    def _act(self, observation):
        return self.actor.get_action(observation)

    def _learn(self, **kwargs):

        #  get batch and unpack it
        #  TODO can I explode this somehow?
        batch = self.memory.get_batch(self.batch_size)
        obs = batch['observations']
        actions = batch['actions']
        rews = batch['rewards']
        next_obs = batch['next_observations']
        terminal = batch['terminal']

        #  create a Bellman target to update our critic
        #  get an estimate of next state value using target network
        #  we use the target network to generate actions
        t_actions = self.actor.get_target_action(next_obs)

        if hasattr(self, 'action_processor'):
            t_actions = self.action_processor.transform(t_actions)

        #  obs is already processed as we pull it from memory
        #  TODO this could be a test!
        #  this is the critics estimate of the return from the
        #  experienced next state, taking actions as suggested by the
        #  actor network
        q_next_obs = self.critic.predict_target(next_obs, t_actions)

        #  set terminal next state Q to zero
        q_next_obs[terminal] = 0

        #  create the Bellman target
        targets = rews + self.discount * q_next_obs

        #  scale the targets
        if hasattr(self, 'target_processor'):
            targets = self.target_processor.transform(targets)

        #  update the critic
        #  actions are already scaled as they are pulled from memory
        #  TODO test
        self.critic.improve(obs, actions, targets)

        #  update the actor
        #  get the actions the actor would take for these observations
        #  using online network (not target)
        actor_actions = self.actor.get_action(obs)

        #  get the gradients for these actions
        act_grads = self.critic.get_action_grads(obs, actor_actions)[0]
        self.actor.improve(obs, act_grads)

        return {}

class DPGActor(object):
    """
    A neural network based acting policy

    Agent maps observation to a determinstic action, then noise is added

    args

    """
    def __init__(self,
                 sess,
                 observation_shape,
                 action_shape,
                 action_space,
                 layers,
                 learning_rate,
                 tau,
                 wt_init=tf.truncated_normal,
                 b_init=tf.zeros,
                 scope='DPG_Actor',
                 **kwargs):

        self.sess = sess
        self.obs_shape = observation_shape
        self.action_shape = action_shape
        self.action_space = action_space

        self.layers = layers
        self.learning_rate = learning_rate
        self.tau = tau

        #  exploration noise
        self.actor_noise = OUActionNoise(np.zeros(*self.action_shape))

        with tf.variable_scope(scope):

            #  actor network
            with tf.variable_scope('actor_online_net'):
                self.obs, self.action, self.ao_sum = self.make_acting_graph(wt_init, b_init)
                self.o_params = self.get_tf_params('{}/actor_online_net'.format(scope))

            #  target network
            with tf.variable_scope('actor_target_net'):
                self.t_obs, self.t_action, self.at_sum = self.make_acting_graph(wt_init, b_init)
                self.t_params = self.get_tf_params('{}/actor_target_net'.format(scope))

            #  tf machinery to improve actor network
            with tf.variable_scope('learning'):
                self.learn_sum = self.make_learning_graph()

            #  ops to update target network
            with tf.variable_scope('target_net_update'):
                self.update_target_net = self.make_target_net_update_ops()

    def get_tf_params(self, name_start):
        params = [p for p in tf.trainable_variables()
                  if p.name.startswith(name_start)]

        return sorted(params, key=lambda var: var.name)

    def make_target_net_update_ops(self):
        """
        Creates the Tensorflow operations to update the target network.

        The two lists of Tensorflow Variables (one for the online net, one
        for the target net) are iterated over together and new weights
        are assigned to the target network
        """
        update_ops = []
        for online, target in zip(self.o_params, self.t_params):
            logging.debug('copying {} to {}'.format(online.name,
                                                    target.name))
            val = tf.add(tf.multiply(online, self.tau),
                         tf.multiply(target, 1 - self.tau))

            operation = target.assign(val)
            update_ops.append(operation)

        return update_ops

    def make_acting_graph(self, wt_init, b_init):
        """
        """
        obs = tf.placeholder(tf.float32, (None, *self.obs_shape), 'obs')

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
            action = tf.layers.dense(inputs=layer,
                                     units=self.action_shape[0],
                                     activation=None)

        #  create summaries for tensorboard
        sums = tf.summary.merge([tf.summary.histogram('observation', obs),
                                 tf.summary.histogram('determ_action', action)])

        return obs, action, sums

    def make_learning_graph(self):
        """
        Used to update the actor graph by applying gradients to
        self.network_param
        """
        #  this gradient is provided by the critic
        self.action_gradient = tf.placeholder(tf.float32,
                                              (None, *self.action_shape))

        #  combine the gradients
        self.actor_gradients = tf.gradients(self.action,
                                            self.o_params,
                                            -self.action_gradient)

        #  clip using global norm
        #self.actor_gradients = tf.clip_by_global_norm(actor_gradients, 5)

        #  define the optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        #  improve the actor network using the DPG algorithm
        self.train_op = self.optimizer.apply_gradients(zip(self.actor_gradients,
                                                           self.o_params))

        #  create summaries for tensorboard
        sums = tf.summary.merge([tf.summary.histogram('action_gradient',
                                                      self.action_gradient)])
                                 # tf.summary.histogram('actor_grads',
                                 #                      self.actor_gradients)])
        return sums

    def get_action(self, obs):
        #  generating an action from the policy network
        #  TODO check what hapens if noise and batch size are different!!!
        determ_action = self.sess.run(self.action, {self.obs: obs})
        noise = self.actor_noise().reshape(-1, *self.action_shape)
        action = determ_action + self.actor_noise()

        #  clipping the action
        lows = np.array([space.low for space in self.action_space.spaces])
        highs = np.array([space.high for space in self.action_space.spaces])
        action = np.clip(action, lows, highs)

        logger.debug('determinsitic_action {}'.format(determ_action))
        logger.debug('noise {}'.format(noise))
        logger.debug('action {}'.format(action))

        return np.array(action).reshape(obs.shape[0], *self.action_shape)

    def get_target_action(self, obs):

        action = self.sess.run(self.t_action, {self.t_obs: obs})

        return np.array(action).reshape(-1, *self.action_shape)

    def improve(self,
                obs,
                act_grads):

        self.sess.run(self.train_op, {self.obs: obs,
                                      self.action_gradient: act_grads})


class DPGCritic(object):
    """
    An on-policy estimate of Q(s,a) for the actor policy

    Single output node Q(s,a)
    """
    def __init__(self,
                 sess,
                 observation_shape,
                 action_shape,
                 layers,
                 learning_rate,
                 tau,
                 wt_init=tf.truncated_normal,
                 b_init=tf.zeros,
                 scope='DPG_Critic',
                 **kwargs):

        self.sess = sess
        self.obs_shape = observation_shape
        self.action_shape = action_shape
        self.layers = layers
        self.learning_rate = learning_rate
        self.tau = tau

        with tf.variable_scope(scope):
            with tf.variable_scope('online_net'):
                self.obs, self.action, self.pred = self.make_pred_graph(wt_init, b_init)
                self.o_params = self.get_tf_params('{}/online_net'.format(scope))

            with tf.variable_scope('target_net'):
                self.t_obs, self.t_action, self.t_pred = self.make_pred_graph(wt_init, b_init)
                self.t_params = self.get_tf_params('{}/target_net'.format(scope))

            with tf.variable_scope('learning'):
                self.make_learning_graph()

            with tf.variable_scope('target_net_update'):
                self.update_target_net = self.make_target_net_update_ops()

    def get_tf_params(self, name_start):
        params = [p for p in tf.trainable_variables()
                  if p.name.startswith(name_start)]

        return sorted(params, key=lambda var: var.name)

    def make_target_net_update_ops(self):
        """
        Creates the Tensorflow operations to update the target network.

        The two lists of Tensorflow Variables (one for the online net, one
        for the target net) are iterated over together and new weights
        are assigned to the target network
        """
        update_ops = []
        for online, target in zip(self.o_params, self.t_params):
            logging.debug('copying {} to {}'.format(online.name,
                                                    target.name))
            val = tf.add(tf.multiply(online, self.tau),
                         tf.multiply(target, 1 - self.tau))

            operation = target.assign(val)
            update_ops.append(operation)

        return update_ops

    def make_pred_graph(self, wt_init, bias_init):

        obs = tf.placeholder(tf.float32,
                             (None, *self.obs_shape),
                             'obs')

        act = tf.placeholder(tf.float32,
                             (None, *self.action_shape),
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
            l1_W = tf.Variable(wt_init([self.layers[0], self.layers[1]]))
            l2_W = tf.Variable(wt_init([*self.action_shape, self.layers[1]]))
            l2_b = tf.Variable(bias_init([self.layers[1]]))

            net = tf.matmul(net, l1_W) + tf.matmul(act, l2_W) + l2_b
            net = tf.layers.batch_normalization(net, training=True)

        #  iterate over self.layers
        for i, nodes in enumerate(self.layers[2:]):
            with tf.variable_scope('hidden_layer_{}'.format(i)):
                net = tf.layers.dense(inputs=net, units=nodes)
                net = tf.layers.batch_normalization(net, training=True)
                net = tf.nn.relu(net)

        #  use a linear layer as the output layer
        #  hardcoded with a single output node
        #  took out the weight inits
        with tf.variable_scope('output_layer'):
            out = tf.layers.dense(inputs=net,
                                  units=1,
                                  activation=None)
        return obs, act, out

    def make_learning_graph(self):
        #  a Bellman target created externally to this value function
        self.target = tf.placeholder(tf.float32, [None, 1], 'target')

        #  add the optimizer, loss function & train operation
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.error = self.target - self.pred
        loss = tf.losses.huber_loss(self.target, self.pred)
        self.train_op = self.optimizer.minimize(loss)

        #  gradient of network parameters wrt action 
        #  this is using online network - CHECK
        self.action_grads = tf.gradients(self.pred, self.action)

    def predict(self, obs, action):
        return self.sess.run(self.pred, {self.obs: obs,
                                         self.action: action})

    def predict_target(self,  obs, action):
        return self.sess.run(self.t_pred, {self.t_obs: obs,
                                           self.t_action: action})

    def improve(self, obs, action, target):
        self.sess.run([self.error, self.train_op],
                      {self.obs: obs,
                       self.action: action,
                       self.target: target})

    def get_action_grads(self, obs, action):
        return self.sess.run(self.action_grads, {self.obs: obs,
                                                 self.action: action})


# Taken from https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OUActionNoise(object):
    """
    OrnsteinUhlenbeckActionNoise
    """
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
