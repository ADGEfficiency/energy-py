"""
Implementation of the Deep Q Network (DQN) algorithm.

Q-learning with experience replay and a target network.

Reference - Mnih et. al 2015

class Agent - the learner and decision maker.
class EpsilonDecayer - decays epsilon used in the e-greedy policy.
class Qfunc - an approximation of Q(s,a) using a Tensorflow feedforward net.

"""

import logging
from random import random as random_uniform

import numpy as np
import tensorflow as tf

from energy_py.agents import BaseAgent
from energy_py import Normalizer


logging.getLogger(__name__)


class DQN(BaseAgent):
    """
    The learner and decision maker.

    All calls to tensorflow are wrapped into methods.

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
    """
    def __init__(self,
                 env,
                 discount,
                 tau,
                 sess,
                 total_steps,
                 batch_size,
                 layers,
                 learning_rate,
                 initial_random=0.1,
                 epsilon_decay_fraction=0.5,
                 memory_fraction=0.25,
                 process_observation=False,
                 process_target=False,
                 act_path=None,
                 learn_path=None,
                 **kwargs):

        self.env = env
        self.discount = discount
        self.tau = tau
        self.sess = sess
        self.batch_size = batch_size

        memory_length = int(total_steps * memory_fraction)
        self.initial_random = initial_random

        #  number of steps where epsilon is decayed from 1.0 to 0.1
        decay_steps = total_steps * epsilon_decay_fraction
        self.epsilon_getter = EpsilonDecayer(decay_steps,
                                             init_random=self.initial_random)

        #  the counter is stepped up every time we act or learn
        self.counter = 0

        self.actions = self.env.discretize(num_discrete=20)
        print('{} actions'.format(self.actions))
        super().__init__(env, discount, memory_length)

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

        if process_observation:
            self.observation_processor = Normalizer()

        if process_target:
            self.target_processor = Normalizer()

        self.acting_writer = tf.summary.FileWriter(act_path,
                                                   graph=self.sess.graph)

        self.learning_writer = tf.summary.FileWriter(learn_path,
                                                     graph=self.sess.graph)

        sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def __repr__(self): return '<class DQN Agent>'

    def make_target_net_update_ops(self):
        """
        Creates the Tensorflow operations to update the target network.

        The two lists of Tensorflow Variables (one for the online net, one
        for the target net) are iterated over together and new weights
        are assigned to the target network
        """
        with tf.variable_scope('update_target_network'):
            update_ops = []
            for online, target in zip(self.online.params, self.target.params):
                logging.debug('copying {} to {}'.format(online.name,
                                                        target.name))
                val = tf.add(tf.multiply(online, self.tau),
                             tf.multiply(target, 1 - self.tau))

                operation = target.assign(val)
                update_ops.append(operation)
        return update_ops

    def remember(self, observation, action, reward, next_observation, done):
        """
        Store experience in the agent's memory.

        args
            observation (np.array)
            action (np.array)
            reward (np.array)
            next_observation (np.array)
            done (np.array)
        """
        observation = observation.reshape(-1, *self.obs_shape)
        next_observation = next_observation.reshape(-1, *self.obs_shape)


        if hasattr(self, 'observation_processor'):
            observation = self.observation_processor.transform(observation)
            next_observation = self.observation_processor.transform(next_observation)

        return self.memory.remember(observation, action, reward,
                                    next_observation, done)

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
        self.learning_writer.add_summary(summary, self.counter)

        logging.debug('predict_target - next_obs {}'.format(observations))
        logging.debug('predict_target - q_vals {}'.format(q_vals))
        logging.debug('predict_target - max_q {}'.format(max_q))

        return max_q.reshape(observations.shape[0], 1)

    def predict_online(self, observation):
        """
        We use our online network to choose actions.

        args
            observation (np.array) a single observation

        returns
            action
        """
        obs = observation.reshape((1, *self.env.observation_space.shape))

        fetches = [self.online.q_values,
                   self.online.max_q,
                   self.online.optimal_action_idx, self.online.acting_summary]

        feed_dict = {self.online.observation: obs}
        q_values, max_q, action_idx, summary = self.sess.run(fetches, feed_dict)
        self.acting_writer.add_summary(summary, self.counter)

        max_q = max_q.flatten()[0]
        max_q_sum = tf.Summary(value=[tf.Summary.Value(tag='max_q_acting',
                                                       simple_value=max_q)])

        self.acting_writer.add_summary(max_q_sum, self.counter)
        self.acting_writer.flush()

        #  index at zero because TF returns an array
        action = self.actions[action_idx[0]]

        logging.debug('predict_online - observation {}'.format(obs))
        logging.debug('predict_online - pred_q_values {}'.format(q_values))
        logging.debug('predict_online - max_q {}'.format(max_q))
        logging.debug('predict_online - action_index {}'.format(action_idx))
        logging.debug('predict_online - action {}'.format(action))

        return action

    def update_target_network(self):
        """
        Updates the target network weights using the parameter tau

        Relies on the sorted lists of tf.Variables kept in each Qfunc object
        """
        logging.debug('updating target net at count {}'.format(self.counter))

        return self.sess.run(self.update_ops)

    def act(self, observation):
        """
        Our agent attempts to manipulate the world.

        Acting according to epsilon greedy policy.

        args
            observation (np.array)

        returns
            action (np.array)
        """
        self.counter += 1
        epsilon = self.epsilon_getter.epsilon
        logging.debug('epsilon is {}'.format(epsilon))

        if epsilon > random_uniform():
            action = self.env.action_space.sample_discrete()
            logging.debug('acting randomly - action is {}'.format(action))
        else:
            action = self.predict_online(observation)
            logging.debug('acting optimally action is {}'.format(action))

        epsilon_sum = tf.Summary(value=[tf.Summary.Value(tag='epsilon', simple_value=epsilon)])
        self.acting_writer.add_summary(epsilon_sum, self.counter)
        self.acting_writer.flush()

        return np.array(action).reshape(1, *self.action_shape)
        # return action

    def learn(self):
        """
        Our agent attempts to make sense of the world.

        A batch sampled using experience replay is used to train the online
        network using targets from the target network.

        returns
            train_info (dict)
        """
        batch = self.memory.get_batch(self.batch_size)
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminal']
        next_observations = batch['next_observations']

        next_obs_q = self.predict_target(next_observations)

        #  if next state is terminal, set the value to zero
        next_obs_q[terminals] = 0

        #  creating a target for Q(s,a) using the Bellman equation
        rewards = rewards.reshape(rewards.shape[0], 1)
        target = rewards + self.discount * next_obs_q

        if hasattr(self, 'target_processor'):
            target = self.target_processor.transform(target)
            logging.info('target_processor_mins {}'.format(self.target_processor.mins))
            logging.info('target_processor_maxs {}'.format(self.target_processor.maxs))

        indicies = np.zeros((actions.shape[0], 1), dtype=int)

        for arr, action in zip(indicies, actions):
            idx = self.actions.index(action.tolist())
            arr[0] = idx

        rng = np.arange(actions.shape[0]).reshape(actions.shape[0], 1)
        indicies = np.concatenate([rng, indicies], axis=1)

        fetches = [self.online.q_values,
                   self.online.q_value,
                   self.online.loss,
                   self.online.train_op,
                   self.online.learning_summary]

        feed_dict = {self.online.observation: observations,
                     self.online.action: indicies,
                     self.online.target: target}

        q_vals, q_val, loss, train_op, train_sum = self.sess.run(fetches, feed_dict)

        logging.debug('learning - observations {}'.format(observations))

        logging.debug('learning - rewards {}'.format(rewards))
        logging.debug('learning - terminals {}'.format(terminals))
        logging.debug('learning - next_obs_q {}'.format(next_obs_q))

        logging.debug('learning - actions {}'.format(actions))
        logging.debug('learning - indicies {}'.format(indicies))
        logging.debug('learning - q_values {}'.format(q_vals))
        logging.debug('learning - q_value {}'.format(q_val))

        logging.debug('learning - target {}'.format(target))
        logging.debug('learning - loss {}'.format(loss))

        self.learning_writer.add_summary(train_sum, self.counter)

        self.update_target_network()

        return {'loss': loss}


class EpsilonDecayer(object):
    """
    A class to decay epsilon.  Epsilon is used in e-greedy action selection.

    Initially act totally random, then linear decay to a minimum.

    Two counters are used
        self.count is the total number of steps the object has seen
        self.decay_count is the number of steps in the decay period

    args
        decay_length (int) len of the linear decay period
        init_random (int) num steps to act fully randomly at start
        eps_start (float) initial value of epsilon
        eps_end (float) final value of epsilon
    """

    def __init__(self,
                 decay_length,
                 init_random=0,
                 eps_start=1.0,
                 eps_end=0.1):

        self.decay_length = int(decay_length)
        self.init_random = int(init_random)
        self.min_start = self.init_random + self.decay_length

        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)

        eps_delta = self.eps_start - self.eps_end
        self.coeff = - eps_delta / self.decay_length

        self.reset()

    def __repr__(self): return '<class Epislon Greedy>'

    def reset(self):
        self.count = 0
        self.decay_count = 0

    @property
    def epsilon(self):
        #  move the counter each step
        self.count += 1

        if self.count <= self.init_random:
            self._epsilon = 1.0

        if self.count > self.init_random and self.count <= self.min_start:
            self._epsilon = self.coeff * self.decay_count + self.eps_start
            self.decay_count += 1

        if self.count > self.min_start:
            self._epsilon = self.eps_end

        return float(self._epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)


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
        print('making tf graph for {}'.format(self.scope))
        print('input shape {}'.format(input_shape))
        print('output shape {}'.format(output_shape))
        print('layers {}'.format(layers))

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

            #  no activation function on the output layer (i.e. linear)
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

            self.error = self.target - self.q_value
            self.loss = tf.losses.huber_loss(self.target, self.q_value)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        #  averages across the batch (ie a scalar to represent the whole batch)
        average_q_val = tf.reduce_mean(self.q_value)

        acting_sum = [tf.summary.histogram('observation_act', self.observation),
                      tf.summary.histogram('input_weights', w1),
                      tf.summary.histogram('input_bias', b1),
                      tf.summary.histogram('output_weights', wout),
                      tf.summary.histogram('output_bias', bout),
                      tf.summary.histogram('q_values', self.q_values),
                      tf.summary.histogram('max_q', self.max_q)]

        self.acting_summary = tf.summary.merge(acting_sum)

        learn_sum = [tf.summary.histogram('observation_learn', self.observation),
                     tf.summary.histogram('q_values', self.q_values),
                     tf.summary.histogram('max_q', self.max_q),
                     tf.summary.histogram('target', self.target),
                     tf.summary.scalar('avg_batch_q_value', average_q_val),
                     tf.summary.histogram('error', self.error),
                     tf.summary.scalar('loss', self.loss)]

        self.learning_summary = tf.summary.merge(learn_sum)
