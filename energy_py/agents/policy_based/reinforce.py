"""
This script contains an agent based on the REINFORCE algorithm.
ref = ???

REINFORCE is a Monte Carlo policy gradient algorithm.

REINFORCE is high variance (due to the nature of Monte Carlo
sampling actual experience).

REINFORCE is a low bias algorithm as we don't bootstrap.

This algorithm requires lots of episodes to run:
- policy gradient only makes small updates
- Monte Carlo is high variance (so takes a while for the expectation to converge)
- we only update once per episode

"""

import numpy as np
import tensorflow as tf

from energy_py.agents.agent_core import Base_Agent
from energy_py.agents.tensorflow_machinery import fc_layer


class REINFORCE_Agent(Base_Agent):
    """
    REINFORCE agent.

    Able to control over a single continuous action space.
    """
    def __init__(self, env,
                       epsilon_decay_steps,
                       learning_rate = 0.01,
                       batch_size    = 64):

        #  passing the environment to the Base_Agent class
        super().__init__(env, epsilon_decay_steps)

        self.learning_rate   = learning_rate
        self.batch_size      = batch_size

        #  initialize the TensorFlow machinery
        with tf.name_scope('policy_network'):
            #  create placeholder variable for the observation
            self.observation = tf.placeholder(tf.float32,
                                              [None, self.observation_dim],
                                              'observation')
            #  create the tensorflow graph, loss & training steps
            self.make_graph()

    def make_graph(self):
        """
        Makes the TensorFlow machinery for a Gaussian Policy Network.

        Code will only work for a single action - needs work.
        """
        #  the TensorFlow graph for the policy network
        output_dim = 2 * self.num_actions  # one for mean, one for standard deviation

        with tf.variable_scope('prediction'):
            #  make a three layer fully-connected neural network
            with tf.variable_scope('input_layer'):
                input_layer = fc_layer(self.observation, [self.observation_dim, self.observation_dim], [self.observation_dim], tf.nn.relu)

            with tf.variable_scope('hidden_layer_1'):
                hidden_layer_1 = fc_layer(input_layer, [self.observation_dim, self.observation_dim * 2], [self.observation_dim*2], tf.nn.relu)

            with tf.variable_scope('hidden_layer_2'):
                hidden_layer_2 = fc_layer(hidden_layer_1, [self.observation_dim*2, self.observation_dim*2], [self.observation_dim*2], tf.nn.relu)

            #with tf.variable_scope('hidden_layer_3'):
                #hidden_layer_3 = fc_layer(hidden_layer_2, [self.obser, 2000], [2000], tf.nn.relu)

            with tf.variable_scope('output_layer'):
                self.output_layer = fc_layer(hidden_layer_2, [self.observation_dim*2, output_dim], [output_dim])

            #  parameterizing normal distributions
            #  indexes for the output layer
            mean_idx = tf.range(start=0, limit=output_dim, delta=2)
            stdev_idx = tf.range(start=1, limit=output_dim, delta=2)

            #  gather ops
            means = tf.gather(params=self.output_layer, indices=mean_idx, axis=1)
            stdevs = tf.gather(params=self.output_layer, indices=stdev_idx, axis=1)

            #  clip the stdev so that stdev >= a small positive number
            stdevs = tf.clip_by_value(stdevs, 1e-10, tf.reduce_max(stdevs))
            self.norm_dist = tf.contrib.distributions.Normal(loc=means, scale=stdevs)

            #  selecting an action by sampling from the distribution
            self.action = self.norm_dist.sample(1)

            #  clipping the action
            lows = np.array([space.low for space in self.action_space])
            highs = np.array([space.high for space in self.action_space])
            self.action = tf.clip_by_value(self.action, lows, highs)

        #  using the score function to calculate the loss
        with tf.variable_scope('learning'):
            self.taken_action = tf.placeholder(tf.float32, [None, self.num_actions], name='taken_actions')
            self.discounted_return = tf.placeholder(tf.float32, [None, 1], 'discounted_returns')

            self.probs = self.norm_dist.prob(self.taken_action)
            self.probs_clipped = tf.clip_by_value(self.probs, 1e-10, 1)
            self.log_probs = tf.log(self.probs_clipped)

            #  we make use of the fact that multiply broadcasts here
            #  discounted returns is of shape (samples, 1)
            #  while log_probs is of shape (samples, num_actions)
            self.loss = tf.multiply(self.log_probs, self.discounted_return)
            self.loss = -tf.reduce_mean(self.loss)

            #  creating the training step
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

        return None

    def _act(self, observation, session, epsilon):
        """
        Main function for agent to take an action

        Could put this code in act() but will require calling the greedy_action
        and random_action methods from the base class

        Maybe greedy and random actions can be part of base class...
        """
        if np.random.uniform() > epsilon:
            #  acting greedily
            action = self.greedy_action(observation, session)
            if self.verbose > 0:
                print('acting greedy with action {}'.format(action))
        else:
            #  print acting randomly
            action = self.random_action()
            if self.verbose > 0:
                print('acting random with action {}'.format(action))
        return action

    def greedy_action(self, observation, session):
        """
        Helper function for _act

        Can probably go into core agent
        """

        #  scaling the observation for use in the policy network
        scaled_observation = self.memory.scale_array(observation,
                                                     self.observation_space)

        scaled_observation = scaled_observation.reshape(-1, self.observation_dim)
        assert scaled_observation.shape[0] == 1

        #  generating an action from the policy network
        action = session.run(self.action, {self.observation : scaled_observation})
        action = action.reshape(self.num_actions)
        return action

    def random_action(self):
        """
        Helper function for _act
        Can probably go into core agent
        """
        if self.verbose > 0:
            print('acting randomly')

        #  sampling the space object for every dimension of the action space
        random_actions = [space.sample() for space in self.action_space]

        #  converting the list to a np array of the correct shape
        action = np.array(random_actions).reshape(self.num_actions)
        assert len(self.action_space) == action.shape[0]
        return action

    def _learn(self, observations, actions, discounted_returns, session):

        feed_dict = {self.observation : observations,
                     self.taken_action : actions,
                     self.discounted_return : discounted_returns}

        _, loss = session.run([self.train_step, self.loss], feed_dict)
        self.memory.losses.append(loss)

        print('loss is {} - discounted returns were'.format(loss, np.sum(discounted_returns)))
        return loss
