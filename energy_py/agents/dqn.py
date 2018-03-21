"""
Implementation of the Deep Q Network (DQN) algorithm.

Q-learning with experience replay and a target network.

Reference - Mnih et. al 2015

class Agent - the learner and decision maker.
class EpsilonDecayer - decays epsilon used in the e-greedy policy.
class Qfunc - an approximation of Q(s,a) using a Tensorflow feedforward net.

Most functionality occurs within the DQN class.  Qfunc initializes the
tensorflow graph but all sess.run's are within the DQN agent class.
"""

import logging
from random import random
import pdb
import numpy as np
import tensorflow as tf

from energy_py.agents import BaseAgent
from energy_py import LinearScheduler


logger = logging.getLogger(__name__)


class DQN(BaseAgent):
    """
    energy_py implementation of DQN - aka Q-Learning with experience
    replay and a target network.

    args
        env (object) either gym or energy_py environment
        discount (float) aka gamma
        tau (float) controls target network update
        sess (tf.Session)
        total_steps (int) over agent lifetime
        batch_size (int)
        layers (tuple)
        learning_rate (float)
        epsilon_decay_fraction (float) % of total steps to decay epsilon over
        memory_fraction (float) as a % of total steps
        process_observation (bool) TODO
        process_target (bool)

    references
        Minh et. al (2015)

    """
    def __init__(self,
                 sess,
                 env,
                 discount,
                 tau,
                 total_steps,
                 batch_size,
                 layers,
                 learning_rate,
                 double_q=False,
                 initial_random=0.1,
                 epsilon_decay_fraction=0.5,
                 memory_fraction=0.25,
                 observation_processor=None,
                 action_processor=None,
                 target_processor=None,
                 **kwargs):

        self.sess = sess
        self.tau = tau
        self.batch_size = batch_size
        self.double_q = double_q
        memory_length = int(total_steps * memory_fraction)

        super().__init__(env, discount, memory_length, total_steps, **kwargs)

        eps_schd_args = {'pre_step': initial_random*total_steps,
                         'sched_step': epsilon_decay_fraction*total_steps,
                         'initial': 1.0,
                         'final': 0.05}

        logger.debug('epsilon sched args {}'.format(eps_schd_args))
        self.epsilon = LinearScheduler(**eps_schd_args)

        self.actions = self.env.discretize(num_discrete=20)
        logger.debug('actions list is {}'.format(self.actions))

        model_config = {'input_shape': self.obs_shape,
                        'output_shape': (len(self.actions),),
                        'layers': layers,
                        'learning_rate': learning_rate}

        #  the two approximations of Q(s,a)
        #  use the same config dictionary for both
        self.online = Qfunc(model_config, scope='online')
        self.target = Qfunc(model_config, scope='target')

        #  set up the operations to copy the online network parameters to
        #  the target network
        self.update_ops = self.make_target_net_update_ops()

        sess.run(tf.global_variables_initializer())

        self.update_target_network(tau=1.0)

    def __repr__(self): return '<class DQN Agent>'

    def make_target_net_update_ops(self):
        """
        Creates the Tensorflow operations to update the target network.

        The two lists of Tensorflow Variables (one for the online net, one
        for the target net) are iterated over together and new weights
        are assigned to the target network
        """
        with tf.variable_scope('update_target_network'):

            self.tf_tau = tf.placeholder(tf.float32,
                                 shape=(),
                                 name='tau')

            update_ops = []
            for online, target in zip(self.online.params, self.target.params):

                o_name, t_name = online.name.split('/')[1:], target.name.split('/')[1:]
                print('copying {} to {}'.format(o_name, t_name))

                logger.debug('copying {} to {}'.format(online.name,
                                                        target.name))

                assert o_name == t_name 
                val = tf.add(tf.multiply(online, self.tf_tau),
                             tf.multiply(target, 1 - self.tf_tau))

                operation = target.assign(val)
                update_ops.append(operation)

        return update_ops

    def predict_target(self, observations):
        """
        Target network is used to predict the maximum discounted expected
        return for the next_observation as experienced by the agent

        args
            observations (np.array)

        returns
            max_q (np.array) shape=(batch_size, 1)
        """
        fetches = [self.target.q_values,
                   self.target.max_q,
                   self.target.acting_summary]

        feed_dict = {self.target.observation: observations}

        q_vals, max_q, summary = self.sess.run(fetches, feed_dict)

        if hasattr(self, 'learning_writer'):
            self.learning_writer.add_summary(summary, self.counter)

        logger.debug('predict_target - next_obs {}'.format(observations))
        logger.debug('predict_target - q_vals {}'.format(q_vals))
        logger.debug('predict_target - max_q {}'.format(max_q))

        return q_vals, max_q.reshape(observations.shape[0], 1)

    def predict_online(self, observation):
        """
        We use our online network to choose actions.

        args
            observation (np.array) a single observation

        returns
            action
        """
        obs = observation.reshape((-1, *self.env.observation_space.shape))

        fetches = [self.online.q_values,
                   self.online.max_q,
                   self.online.optimal_action_idx, self.online.acting_summary]

        feed_dict = {self.online.observation: obs}
        q_vals, max_q, action_idx, summary = self.sess.run(fetches, feed_dict)

        max_q = max_q.flatten()[0]
        max_q_sum = tf.Summary(value=[tf.Summary.Value(tag='max_q_acting',
                                                       simple_value=max_q)])

        if hasattr(self, 'learning_writer'):
            self.acting_writer.add_summary(summary, self.counter)
            self.acting_writer.add_summary(max_q_sum, self.counter)
            self.acting_writer.flush()

        q_vals = q_vals.reshape(obs.shape[0], len(self.actions))

        #  create a tiled array of actions
        tiled = np.tile(np.array(self.actions),
                        obs.shape[0]).reshape(obs.shape[0],
                                              len(self.actions),
                                              *self.action_shape)

        #  index out the action
        action = tiled[np.arange(obs.shape[0]), action_idx, :]
        action = np.array(action).reshape(obs.shape[0], *self.action_shape)

        logger.debug('predict_online - observation {}'.format(obs))
        logger.debug('predict_online - pred_q_values {}'.format(q_vals))
        logger.debug('predict_online - max_q {}'.format(max_q))
        logger.debug('predict_online - action_index {}'.format(action_idx))
        logger.debug('predict_online - action {}'.format(action))

        return q_vals, action_idx, action

    def update_target_network(self, tau=None):
        """
        Updates the target network weights using the parameter tau

        Relies on the sorted lists of tf.Variables kept in each Qfunc object
        """
        if tau is None:
            tau = self.tau
        logger.debug('updating target net count {} tau {}'.format(self.counter,
                                                                  tau))

        self.sess.run(self.update_ops, {self.tf_tau: tau})

    def _act(self, observation):
        """
        Our agent attempts to manipulate the world.

        Acting according to epsilon greedy policy.

        args
            observation (np.array)

        returns
            action (np.array)
        """
        self.counter += 1
        eps = self.epsilon()
        logger.debug('epsilon is {}'.format(eps))

        if eps > random():
            action = self.env.sample_discrete()
            logger.debug('acting randomly - action is {}'.format(action))
        else:
            _, _, action = self.predict_online(observation)
            logger.debug('acting optimally action is {}'.format(action))

        if hasattr(self, 'acting_writer'):
            epsilon_sum = tf.Summary(value=[tf.Summary.Value(tag='epsilon',
                                                             simple_value=eps)])
            self.acting_writer.add_summary(epsilon_sum, self.counter)
            self.acting_writer.flush()

        return np.array(action).reshape(1, *self.action_shape)

    def _learn(self):
        """
        Our agent attempts to make sense of the world.

        A batch sampled using experience replay is used to train the online
        network using targets from the target network.

        returns
            train_info (dict)
        """
        #  TODO could just make the other memory types accept beta as an arg
        #  and not use it
        logger.debug('getting batch from memory')

        if self.memory_type == 'priority':
            beta = self.beta()
            batch = self.memory.get_batch(self.batch_size,
                                          beta=beta)

            if hasattr(self, 'acting_writer'):
                beta_sum = tf.Summary(value=[tf.Summary.Value(tag='beta',
                                                              simple_value=beta)])
                self.acting_writer.add_summary(beta_sum, self.counter)
                self.acting_writer.flush()

        else:
            batch = self.memory.get_batch(self.batch_size)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminal']
        next_observations = batch['next_observations']

        #  if we are doing prioritiezed experience replay then our
        #  batch dict will have the importance weights
        #  if not we create an array of ones
        if self.memory_type == 'priority':
            importance_weights = batch['importance_weights']
        else:
            importance_weights = np.ones_like(rewards)

        if self.double_q is False:
            #  the DQN update
            #  max across the target network
            logger.debug('DQN update')
            _, next_obs_q = self.predict_target(next_observations)

        if self.double_q:
            logger.debug('DDQN update')
            #  argmax across the online network to find the action 
            #  the online net thinks is optimal
            _, action_index, _ = self.predict_online(next_observations)

            #  get the predicted Q values for the target network
            t_q_vals, _ = self.predict_target(next_observations)

            next_obs_q = t_q_vals[np.arange(next_observations.shape[0]),
                                  action_index]

        next_obs_q = next_obs_q.reshape(next_observations.shape[0], 1)

        #  if next state is terminal, set the value to zero
        next_obs_q[terminals] = 0

        #  creating a target for Q(s,a) using the Bellman equation
        target = rewards + self.discount * next_obs_q

        if hasattr(self, 'target_processor'):
            target = self.target_processor.transform(target)
            logger.info('target_processor_mins {}'.format(self.target_processor.mins))
            logger.info('target_processor_maxs {}'.format(self.target_processor.maxs))

        indicies = np.zeros((actions.shape[0], 1), dtype=int)

        for arr, action in zip(indicies, actions):
            idx = self.actions.index(action.tolist())
            arr[0] = idx

        rng = np.arange(actions.shape[0]).reshape(actions.shape[0], 1)
        indicies = np.concatenate([rng, indicies], axis=1)

        fetches = [self.online.q_values,
                   self.online.q_value,
                   self.online.loss,
                   self.online.td_errors,
                   self.online.train_op,
                   self.online.learning_summary]

        feed_dict = {self.online.observation: observations,
                     self.online.action: indicies,
                     self.online.target: target,
                     self.online.importance_weights: importance_weights}

        q_vals, q_val, loss, td_errors, train_op, train_sum = self.sess.run(fetches, feed_dict)

        logger.debug('learning - observations {}'.format(observations))
        logger.debug('learning - rewards {}'.format(rewards))
        logger.debug('learning - terminals {}'.format(terminals))
        logger.debug('learning - next_obs_q {}'.format(next_obs_q))
        logger.debug('learning - importance_weights {}'.format(importance_weights))

        logger.debug('learning - actions {}'.format(actions))
        logger.debug('learning - indicies {}'.format(indicies))
        logger.debug('learning - q_values {}'.format(q_vals))
        logger.debug('learning - q_value {}'.format(q_val))

        logger.debug('learning - target {}'.format(target))
        logger.debug('learning - loss {}'.format(loss))
        logger.debug('learning - td_errors {}'.format(td_errors))

        if self.memory_type == 'priority':
            self.memory.update_priorities(batch['indexes'],
                                          td_errors)

        if hasattr(self, 'learning_writer'):
            self.learning_writer.add_summary(train_sum, self.counter)

        self.update_target_network()

        return {'loss': loss}


class Qfunc(object):
    """
    Approximation of the action-value function Q(s,a) using a feedforward
    neural network built in TensorFlow.

    args
        model_config (dict) used to build tf machinery.  see make_graph for args
        scope (str)

    methods
        make_graph(**model_config)

    attributes
        observation
        target
        q_values
        max_q
        optimal_action_idx
        error
        loss
        train_op
        acting_summary
        learning_summary
    """
    def __init__(self, model_config, scope):
        self.scope = scope

        with tf.variable_scope(scope):
            self.make_graph(**model_config)

        #  tensorflow variables for this model
        #  weights and biases of the neural network
        params = [p for p in tf.trainable_variables()
                  if p.name.startswith(scope)]

        #  sort parameters list by the variable name
        self.params = sorted(params, key=lambda var: var.name)

    def __repr__(self): return '<Q(s,a) {} network>'.format(self.scope)

    def make_graph(self,
                   input_shape,
                   output_shape,
                   layers,
                   learning_rate,
                   w_init=tf.truncated_normal,
                   b_init=tf.zeros,
                   **kwargs):
        """
        Creates all Tensorflow machinery required for acting and learning.

        Could be split into a acting & learning section.

        We never need to train our target network - params for the target
        network are updated by copying weights - done in the DQN agent.

        args
            input_shape (tuple)
            output_shape (tuple)
            layers (list)
            learning_rate (float)
            wt_init (function) tf function used to initialize weights
            b_init (function) tf function used to initialize the biases
        """
        logger.info('making tf graph for {}'.format(self.scope))
        logger.info('input shape {}'.format(input_shape))
        logger.info('output shape {}'.format(output_shape))
        logger.info('layers {}'.format(layers))

        #  aka state - the input to the network
        self.observation = tf.placeholder(tf.float32,
                                          shape=(None, *input_shape),
                                          name='observation')

        #  used to index the network output
        #  first dimension = batch
        #  second dimension = the index of the action
        #  [0, 1] = first batch sample, second action
        #  [4, 0] = 5th sample, first action
        self.action = tf.placeholder(tf.int32,
                                     shape=(None, 2),
                                     name='action')

        #  the target is for the action being trained
        #  shape = (batch_size, 1)
        self.target = tf.placeholder(tf.float32,
                                     shape=(None, 1),
                                     name='target')

        self.importance_weights = tf.placeholder(tf.float32,
                                                 shape=(None, 1),
                                                 name='importance_weights')

        with tf.name_scope('input_layer'):
            #  variables for the input layer weights & biases
            w1 = tf.Variable(w_init([*input_shape, layers[0]]), 'in_w')
            b1 = tf.Variable(b_init(layers[0]), 'in_bias')

            #  construct the layer and use a relu at the end
            layer = tf.add(tf.matmul(self.observation, w1), b1)
            layer = tf.nn.relu(layer)

        for i, nodes in enumerate(layers[1:], 1):
            with tf.variable_scope('hidden_layer_{}'.format(i)):
                w = tf.Variable(w_init([layers[i-1], nodes]), '{}_w'.format(i))
                b = tf.Variable(b_init(nodes), '{}_b'.format(i))
                layer = tf.add(tf.matmul(layer, w), b)
                layer = tf.nn.relu(layer)

        with tf.variable_scope('output_layer'):
            wout = tf.Variable(w_init([nodes, *output_shape]), 'out_w')
            bout = tf.Variable(b_init(*output_shape), 'out_b')

            #  nr activation function on the output layer (i.e. linear)
            self.q_values = tf.add(tf.matmul(layer, wout), bout)

        with tf.variable_scope('argmax'):
            #  ops for selecting optimal action
            max_q = tf.reduce_max(self.q_values, axis=1, name='max_q')
            self.max_q = tf.reshape(max_q, (-1, 1))
            self.optimal_action_idx = tf.argmax(self.q_values, axis=1,
                                                name='optimal_action_idx')

        with tf.variable_scope('learning'):
            #  index out Q(s,a) for the action being trained
            q_value = tf.gather_nd(self.q_values, self.action, name='q_value')
            self.q_value = tf.reshape(q_value, (-1, 1))

            self.td_errors = tf.subtract(self.q_value, self.target)

            self.loss = tf.losses.huber_loss(labels=self.target,
                                             predictions=self.q_value,
                                             weights=self.importance_weights)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        #  averages across the batch (ie a scalar to represent the whole batch)
        average_q_val = tf.reduce_mean(self.q_value)

        # acting_sum = [tf.summary.histogram('observation_act', self.observation),
        #               tf.summary.histogram('input_weights', w1),
        #               tf.summary.histogram('input_bias', b1),
        #               tf.summary.histogram('output_weights', wout),
        #               tf.summary.histogram('output_bias', bout),

        acting_sum = [tf.summary.histogram('q_values', self.q_values),
                      tf.summary.histogram('max_q', self.max_q)]

        self.acting_summary = tf.summary.merge(acting_sum)

        learn_sum = [tf.summary.histogram('observation_learn', self.observation),
                     tf.summary.histogram('q_values', self.q_values),
                     tf.summary.histogram('max_q', self.max_q),
                     tf.summary.histogram('target', self.target),
                     tf.summary.scalar('avg_batch_q_value', average_q_val),
                     tf.summary.histogram('td_error', self.td_errors),
                     tf.summary.scalar('loss', self.loss)]

        self.learning_summary = tf.summary.merge(learn_sum)
