import logging

import numpy as np
import tensorflow as tf

from energypy.agents.agent import BaseAgent
from energypy.common.policies import epsilon_greedy_policy, softmax_policy

from energypy.common.np_utils import find_sub_array_in_2D_array as find_action
from energypy.common.tf_utils import make_copy_ops, get_tf_params
from energypy.common.utils import read_iterable_from_config

from energypy import make_network


logger = logging.getLogger(__name__)


class DQN(BaseAgent):

    def __init__(
            self,
            discount=0.95,
            total_steps=10000,

            double_q=False,
            network='ff',

            filters=None,
            kernels=None,
            strides=None,

            layers=(64, 32, 16),

            num_discrete_actions=20,

            memory_fraction=0.2,

            policy='e_greedy',
            epsilon_decay_fraction=0.3,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            initial_temp=10,
            final_temp=0.5,

            batch_size=64,
            learning_rate=0.001,
            learning_rate_decay=1.0,
            gradient_norm_clip=0.5,

            batch_norm_center=True,
            batch_norm_training=False,
            batch_norm_trainable=True,

            update_target_net=1,
            tau=0.001,
            **kwargs):

        self.total_steps = int(total_steps)

        super().__init__(
            memory_length=self.total_steps*float(memory_fraction),
            **kwargs
        )

        self.double_q = bool(double_q)
        self.batch_norm_center = batch_norm_center
        self.batch_norm_training = batch_norm_training
        self.batch_norm_trainable = batch_norm_trainable

        self.discrete_actions = self.env.action_space.discretize(
            num_discrete_actions)
        self.num_actions = self.discrete_actions.shape[0]

        self.network_id = network
        self.layers = read_iterable_from_config(layers)

        if self.network_id == 'conv':
            self.filters = read_iterable_from_config(filters)
            self.kernels = read_iterable_from_config(kernels)
            self.strides = read_iterable_from_config(strides)

        self.policy = str(policy)
        self.epsilon_decay_fraction = float(epsilon_decay_fraction)
        self.initial_epsilon = float(initial_epsilon)
        self.final_epsilon = float(final_epsilon)
        self.initial_temp = float(initial_temp)
        self.final_temp = float(final_temp)

        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)

        self.update_target_net = int(update_target_net)
        self.tau_val = float(tau)

        with tf.variable_scope('constants'):

            self.discount = tf.Variable(
                initial_value=float(discount),
                trainable=False,
                name='gamma'
            )

            self.discrete_actions_tensor = tf.Variable(
                initial_value=self.discrete_actions,
                trainable=False,
                name='discrete_actions',
            )

            self.gradient_norm_clip = tf.Variable(
                initial_value=float(gradient_norm_clip),
                trainable=False,
                name='gradient_norm_clip'
            )

        with tf.variable_scope('placeholders'):

            self.observation = tf.placeholder(
                shape=(None, *self.env.observation_space.shape),
                dtype=tf.float32,
                name='observation'
            )

            self.selected_action_indicies = tf.placeholder(
                shape=(None),
                dtype=tf.int64,
                name='selected_action_indicies',
            )

            self.reward = tf.placeholder(
                shape=(None),
                dtype=tf.float32,
                name='reward'
            )

            self.next_observation = tf.placeholder(
                shape=(None, *self.env.observation_space.shape),
                dtype=tf.float32,
                name='next_observation'
            )

            self.terminal = tf.placeholder(
                shape=(None),
                dtype=tf.bool,
                name='terminal'
            )

            self.learn_step_tensor = tf.placeholder(
                shape=(),
                dtype=tf.int64,
                name='learn_step_tensor'
            )

            self.explore_toggle = tf.placeholder(
                shape=(),
                dtype=tf.float32,
                name='explore_toggle'
            )

        self.build_acting_graph()

        self.build_learning_graph()

    def build_acting_graph(self):

        with tf.variable_scope('online') as scope:

            self.online_q_values = make_network(
                network_id=self.network_id,

                scope='online_obs',
                input_tensor=self.observation,
                input_shape=self.observation_space.shape,
                layers=self.layers,
                output_nodes=self.num_actions,

                filters=self.filters,
                kernels=self.kernels,
                strides=self.strides
            )

            self.summaries['acting'].extend([
                tf.summary.histogram('acting_q_values', self.online_q_values)
            ])

            if self.double_q:
                scope.reuse_variables()

                self.online_next_obs_q = make_network(
                    network_id=self.network_id,

                    scope='online_next_obs',
                    input_tensor=self.next_observation,
                    input_shape=self.observation_space.shape,
                    layers=self.layers,
                    output_nodes=self.num_actions,

                    filters=self.filters,
                    kernels=self.kernels,
                    strides=self.strides
                )

        with tf.variable_scope('{}_policy'.format(self.policy)):
            if self.policy == 'e_greedy':
                self.epsilon, self.policy = epsilon_greedy_policy(
                    self.online_q_values,
                    self.discrete_actions_tensor,
                    self.learn_step_tensor,
                    self.total_steps * self.epsilon_decay_fraction,
                    self.initial_epsilon,
                    self.final_epsilon,
                    self.explore_toggle
                )

            elif self.policy == 'softmax':
                policy_params = softmax_policy(
                    self.online_q_values,
                    self.online_q_values,
                    self.discrete_actions_tensor,
                    self.learn_step_tensor,
                    self.total_steps,
                    self.initial_temp,
                    self.final_temp
                )

                #  TODO
                self.temp, _, self.log_probs, self.entropy, _, self.policy = policy_params

            else:
                raise ValueError('{} policy not supported'.format(self.policy))

    def build_learning_graph(self):
        with tf.variable_scope('target', reuse=False):

            self.target_q_values = make_network(
                network_id=self.network_id,

                scope='target',
                input_tensor=self.next_observation,
                input_shape=self.observation_space.shape,
                layers=self.layers,
                output_nodes=self.num_actions,

                filters=self.filters,
                kernels=self.kernels,
                strides=self.strides
            )

        self.online_params = get_tf_params('online')
        self.target_params = get_tf_params('target')

        self.copy_ops, self.tau = make_copy_ops(
            self.online_params,
            self.target_params
        )

        with tf.variable_scope('bellman_target'):
            self.q_selected_actions = tf.reduce_sum(
                self.online_q_values * tf.one_hot(
                    self.selected_action_indicies,
                    self.num_actions
                ),
                1
            )

            if self.double_q:
                online_actions = tf.argmax(self.online_next_obs_q, axis=1)

                unmasked_next_state_max_q = tf.reduce_sum(
                    self.target_q_values * tf.one_hot(
                        online_actions, self.num_actions),
                    axis=1,
                    keepdims=True
                )

            else:
                unmasked_next_state_max_q = tf.reduce_max(
                    self.target_q_values,
                    reduction_indices=1,
                    keepdims=True
                )

            self.next_state_max_q = tf.where(
                self.terminal,
                tf.zeros_like(unmasked_next_state_max_q),
                unmasked_next_state_max_q,
                name='terminal_mask'
            )

            self.bellman = self.reward + self.discount * self.next_state_max_q

            """
            batch norm requires some reshaping with a known rank
            reshape the input into batch norm, then flatten in loss
            training=True to normalize each batch
            training=False to use historical statistics
            """

            bellman_norm = tf.layers.batch_normalization(
                tf.reshape(self.bellman, (-1, 1)),
                center=self.batch_norm_center,
                training=self.batch_norm_training,
                trainable=self.batch_norm_trainable
            )

        with tf.variable_scope('optimization'):
            error = tf.losses.huber_loss(
                tf.reshape(bellman_norm, (-1,)),
                self.q_selected_actions,
                scope='huber_loss'
            )

            loss = tf.reduce_mean(error)

            if self.learning_rate_decay:
                self.learning_rate = tf.train.exponential_decay(
                    self.learning_rate,
                    global_step=self.learn_step_tensor,
                    decay_steps=self.total_steps,
                    decay_rate=self.learning_rate_decay,
                    staircase=False,
                    name='learning_rate'
                )

            # optimizer = tf.train.AdamOptimizer(
            #     learning_rate=self.learning_rate
            # )

            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate
            )

            with tf.variable_scope('gradient_clipping'):

                grads_and_vars = optimizer.compute_gradients(
                    loss,
                    var_list=self.online_params
                )

                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[idx] = (tf.clip_by_norm(
                            grad, self.gradient_norm_clip),
                            var
                        )

                        self.summaries['learning'].append(tf.summary.histogram(
                            '{}_gradient'.format(
                                var.name.replace(':', '_')),
                            grad)
                                              )

                self.train_op = optimizer.apply_gradients(grads_and_vars)

        self.summaries['acting'].extend([
            tf.summary.scalar('learning_rate', self.learning_rate),
            tf.summary.scalar('epsilon', self.epsilon),
            tf.summary.scalar('explore_toggle', self.explore_toggle),
                               ])

        self.summaries['acting'].extend([
            tf.summary.histogram(
                self.online_params[-1].name.replace(':', '_'),
                self.online_params[-1]),
            tf.summary.histogram(
                self.online_params[-2].name.replace(':', '_'),
                self.online_params[-2]),

            tf.summary.histogram(
                self.target_params[-1].name.replace(':', '_'),
                self.target_params[-1]),
            tf.summary.histogram(
                self.target_params[-2].name.replace(':', '_'),
                self.target_params[-2]),
                               ])

        self.summaries['learning'].extend([
            tf.summary.histogram('bellman', self.bellman),
            tf.summary.histogram('bellman_norm', bellman_norm),
            tf.summary.scalar('loss', loss),
            tf.summary.histogram('unmasked_next_state_max_q', unmasked_next_state_max_q),
            tf.summary.histogram('next_state_max_q', self.next_state_max_q),
            tf.summary.histogram('target_q_values', self.target_q_values),
                               ])

        self.summaries['acting'] = tf.summary.merge(self.summaries['acting'])
        self.summaries['learning'] = tf.summary.merge(self.summaries['learning'])

        self.sess.run(
            tf.global_variables_initializer()
        )

        #  initialize the target net weights with the online weights
        self.sess.run(
            self.copy_ops,
            {self.tau: 1.0}
        )

    def __repr__(self):
        return '<energypy DQN agent>'

    def _act(self, observation, explore=1.0):
        """ selecting an action based on an observation """
        action, summary = self.sess.run(
            [self.policy, self.summaries['acting']],
            {self.learn_step_tensor: self.learn_step,
             self.explore_toggle: float(explore),
             self.observation: observation}
        )

        self.writers['acting'].add_summary(summary, self.act_step)
        self.writers['acting'].flush()

        logger.debug('observation {}'.format(observation))
        logger.debug('action {}'.format(action))
        logger.debug('learn_step {}'.format(self.learn_step))
        logger.debug('explore {}'.format(explore))

        return action.reshape(1, *self.env.action_space.shape)

    def _learn(self):
        """ our agent attempts to make sense of the world """
        if self.memory.type == 'priority':
            raise NotImplementedError(
                'Add importance sample weights to loss as per pervious version'
            )

        batch = self.memory.get_batch(self.batch_size)

        #  awkward bit - finding the action indicies using np :(
        #  working on a tensorflow solution
        indicies = []
        for action in batch['action']:
            indicies.append(
                find_action(np.array(action).reshape(-1), self.discrete_actions)
            )

        _, summary = self.sess.run(
            [self.train_op, self.summaries['learning']],
            {self.learn_step_tensor: self.learn_step,
             self.observation: batch['observation'],
             self.selected_action_indicies: indicies,
             self.reward: batch['reward'],
             self.next_observation: batch['next_observation'],
             self.terminal: batch['done']  #  should be ether done or terminal TODO
             }
        )

        self.writers['learning'].add_summary(summary, self.learn_step)
        self.writers['learning'].flush()

        if self.learn_step % self.update_target_net == 0:
            _ = self.sess.run(
                self.copy_ops,
                {self.tau: self.tau_val}
            )
