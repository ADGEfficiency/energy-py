"""
todo
- processors
- logging
- tensorboard

"""
import numpy as np
import tensorflow as tf

import energy_py
from energy_py.agents import BaseAgent

from networks import feed_forward, make_copy_ops
from policies import e_greedy
from utils import find_sub_array_in_2D_array as find_action



class DQN(BaseAgent):
    """
    The new energy_py implementation of Deep Q-Network

    BaseAgent args (passed as **kwargs)

    DQN args

    """

    def __init__(
            self,
            discount=0.9,
            total_steps=10000,
            num_discrete_actions=20,
            hiddens=(5, 5, 5),
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_fraction=0.3,
            double_q=False,
            batch_size=64,
            **kwargs):

        super().__init__(**kwargs)

        self.discount = tf.Variable(
            initial_value=discount,
            trainable=False,
            name='gamma')

        self.double_q = double_q
        self.batch_size = batch_size

        self.discrete_actions = self.env.discretize_action_space(
            num_discrete_actions)
        self.num_actions = self.discrete_actions.shape[0]

        #  idea is to have a single step that only starts when learning starts
        #  TODO when does this get increased?
        self.step = tf.Variable(initial_value=0,
                                trainable=False,
                                dtype=tf.int64)

        self.observation = tf.placeholder(
            shape=(None, *self.env.obs_space_shape),
            dtype=tf.float32
        )

        # self.selected_actions = tf.placeholder(
        #     shape=(None, *self.env.action_space_shape), dtype=tf.int64)

        self.selected_action_indicies = tf.placeholder(
            shape=(None), dtype=tf.int64)

        self.reward = tf.placeholder(shape=(None), dtype=tf.float32)

        self.next_observation = tf.placeholder(
            shape=(None, *self.env.obs_space_shape),
            dtype=tf.float32)

        self.terminal = tf.placeholder(shape=(None), dtype=tf.bool)

        with tf.variable_scope('online', reuse=tf.AUTO_REUSE):
            self.online_q_values = feed_forward(
                self.observation,
                hiddens,
                self.num_actions,
                output_activation='linear',
            )

            if self.double_q:
                self.online_next_obs_q = feed_forward(
                    self.next_observation,
                    hiddens,
                    self.num_actions,
                    output_activation='linear',
                 )

        with tf.variable_scope('target', reuse=False):
            self.target_q_values = feed_forward(
                self.next_observation,
                hiddens,
                self.num_actions,
                output_activation='linear',
            )

        from networks import get_tf_params

        self.online_params = get_tf_params('online')
        self.target_params = get_tf_params('target')

        self.copy_ops, self.tau = make_copy_ops(
            self.online_params,
            self.target_params,
        )

        self.discrete_actions_tensor = tf.Variable(
            initial_value=self.discrete_actions,
            trainable=False,
            name='discrete_actions',
        )

        #  would also be possible to send the policy object in from the outside
        self.epsilon, self.policy = e_greedy(
            self.online_q_values,
            self.discrete_actions_tensor,
            self.step,
            total_steps * epsilon_decay_fraction,
            initial_epsilon,
            final_epsilon
        )

        #  now the learning part of the graph

        self.q_selected_actions = tf.reduce_sum(
            self.online_q_values * tf.one_hot(self.selected_action_indicies,
                                              self.num_actions),
            1
        )


        if self.double_q:
            #  online net approximation of the next observation
            #  creating this means we can avoid session calls

            #  the action our online net would take in the next observation
            online_actions = tf.argmax(self.online_next_obs_q, axis=1)

            next_state_max_q = tf.reduce_sum(
                self.target_q_values * tf.one_hot(online_actions,
                                                 self.num_actions),
                1
            )

        else:
            next_state_max_q = tf.reduce_max(
                self.target_q_values,
                reduction_indices=1,
                keepdims=True
            )

        #  masking out the value of the next state for terminal states
        self.next_state_max_q = tf.where(
            self.terminal,
            next_state_max_q,
            tf.zeros_like(next_state_max_q)
        )

        self.bellman = self.reward + self.discount * self.next_state_max_q

        error = tf.losses.huber_loss(
            self.bellman,
            self.q_selected_actions,
            weights=1.0,
            scope='huber_loss'
        )

        #  TODO support for prioritized experience repaly weights
        loss = tf.reduce_mean(error)

        learning_rate = 0.001
        decay_learning_rate = True
        gradient_norm_clip = True

        if decay_learning_rate:
            learning_rate = tf.train.exponential_decay(
                0.01,
                global_step=self.step,
                decay_steps=total_steps,
                decay_rate=0.96,
                staircase=False,
                name='learning_rate'
            )

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if gradient_norm_clip:
            grads_and_vars = optimizer.compute_gradients(loss, var_list=self.online_params)

            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, gradient_norm_clip), var)
            self.train_op = optimizer.apply_gradients(grads_and_vars)

        else:
            self.train_op = optimizer.minimize(loss, var_list=self.online_params)

        #  do stuff with intiailizing vars, and doing the first target net copy op
        self.sess.run(
            tf.global_variables_initializer()
        )

        #  initializing target network weights the same as online network weights
        self.sess.run(
            self.copy_ops,
            {self.tau: 1.0}
        )


    def __repr__(self):
        return '<energy_py DQN agent>'

    def _act(self, observation):

        action = self.sess.run(
            self.policy,
            {self.observation: observation}
        )

        return action.reshape(1, *self.env.action_space_shape)

    def _learn(self):
        """
        Our agent attempts to make sense of the world
        """
        if self.memory_type == 'priority':
            raise NotImplementedError()

        batch = self.memory.get_batch(self.batch_size)

        #  awkward bit - finding the indicies using np :(
        #  working on a solution
        # indicies = [
        #     find_action(np.array(action).reshape(-1), self.discrete_actions)
        #     for action in batch['action']
        # ]
        indicies = []
        for action in batch['action']:
            indicies.append(
                find_action(np.array(action).reshape(-1), self.discrete_actions)
            )

        _ = self.sess.run(
            self.train_op,
            {self.observation: batch['observation'],
             self.selected_action_indicies: indicies,
             self.reward: batch['reward'],
             self.next_observation: batch['next_observation'],
             self.terminal: batch['done']  #  should be ether done or terminal TODO
             }
        )

if __name__ == '__main__':
    env = energy_py.make_env('CartPole')
    obs = env.observation_space.sample()
    discount = 0.95

    with tf.Session() as sess:
        agent = DQN(
            sess=sess,
            env=env,
            total_steps=10,
            discount=discount,
            memory_type='deque',
            learning_rate=1.0
        )
        obs = env.reset()

        for step in range(20):
            act = agent.act(obs)
            next_obs, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, next_obs, done)
            obs = next_obs

        agent.learn()
