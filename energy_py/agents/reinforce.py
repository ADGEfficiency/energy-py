import logging

import numpy as np
import tensorflow as tf

from energy_py import Normalizer, Standardizer
from energy_py.agents import BaseAgent


class GaussianPolicy(object):
    """
    A Gaussian policy approximated using a TensorFlow neural network.

    Mean and standard deviation are parameterized for each action.

    args
        model_dict (dict)

        todo
            how to set the output nodes vs num actions (in agent or in here?)
    """
    def __init__(self, model_dict, scope='GaussianPolicy'):
        logger.info('creating {}'.format(scope))

        for k, v in model_dict.items():
            logger.info('{} : {}'.format(k, v))

        #  network structure
        self.input_nodes = model_dict['input_nodes']
        self.output_nodes = model_dict['output_nodes']
        self.layers = model_dict['layers']

        #  optimizer parameters
        self.lr = model_dict['lr']

        #  action space so we can clip actions
        self.action_space = model_dict['action_space']

        #  create the TensorFlow graphs
        self.action = self.make_acting_graph()
        self.train_op = self.make_learning_graph()

    def make_acting_graph(self):
        #  the TensorFlow graph for the policy network
        #  initialize the TensorFlow machinery
        with tf.name_scope('policy_network'):
            #  create placeholder variable for the observation
            self.obs = tf.placeholder(tf.float32,
                                              [None, self.input_nodes],
                                              'observation')
            obs = tf.reshape(self.obs, (None, self.input_nodes, 1))

            #  add the input layer
            with tf.variable_scope('inputlayer'):

                layer = tf.layers.dense(inputs=self.obs,
                                        units=self.layers[0],
                                        activation=tf.nn.relu)

            #  iterate over self.layers
            for i, nodes in enumerate(self.layers[1:]):
                with tf.variable_scope('input_layer_{}'.format(i)):
                    layer = tf.layers.dense(inputs=layer,
                                            units=nodes,
                                            activation=tf.nn.relu)

            #  return the means and standard deviations for each action
            with tf.variable_scope('output_layer'):
                self.output_layer = tf.layers.dense(inputs=layer,
                                        units=self.output_nodes)

            #  parameterizing normal distributions
            #  one mean & standard deviation per action
            #  as per TRPO paper we parameterize log(standard deviation)
            #  see Schulman et. al (2017) Trust Region Policy Optimization

            #  indexes for the output layer
            mean_idx = tf.range(start=0, limit=self.output_nodes, delta=2)
            stdev_idx = tf.range(start=1, limit=self.output_nodes, delta=2)

            #  gather ops
            self.means = tf.gather(params=self.output_layer, indices=mean_idx, axis=1)
            stdevs = tf.gather(params=self.output_layer, indices=stdev_idx, axis=1)
            self.stdevs = tf.exp(stdevs) + 1e-5
            self.norm_dist = tf.contrib.distributions.Normal(loc=self.means, scale=self.stdevs)

            #  selecting an action by sampling from the distribution
            self.action = self.norm_dist.sample()

            #  clipping the action
            lows = np.array([space.low for space in self.action_space.spaces])
            highs = np.array([space.high for space in self.action_space.spaces])

            self.action = tf.clip_by_value(self.action, lows, highs)

        return self.action

    def make_learning_graph(self):
        with tf.variable_scope('learning'):
            self.taken_action = tf.placeholder(tf.float32, 
                                               [None, self.output_nodes/2],
                                               'taken_actions')
            self.returns = tf.placeholder(tf.float32, [None, 1], 'discounted_returns')

            self.log_probs = self.norm_dist.log_prob(self.taken_action)

            #  we make use of the fact that multiply broadcasts here
            #  discounted returns is of shape (samples, 1)
            #  while log_probs is of shape (samples, num_actions)
            pg_loss = tf.reduce_mean(-self.log_probs * self.returns)

            #  add in some cross entropy cost for exploration
            ce_loss = tf.reduce_mean(1e-1 * self.norm_dist.entropy())
            self.loss = pg_loss - ce_loss

            #  creating the training step
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss)

        return self.train_op

    def get_action(self, session, observation):
        assert observation.shape[0] == 1

        #  generating an action from the policy network
        results = session.run([self.means, self.stdevs, self.action], {self.obs : observation})

        output = {'means' : results[0],
                  'stdevs': results[1],
                  'action': results[2]}

        return output['action'].reshape(int(self.output_nodes/2)), output

    def improve(self, 
                session,
                observations,
                actions,
                returns):

        assert observations.shape[0] == actions.shape[0]
        assert actions.shape[0] == returns.shape[0]

        feed_dict = {self.obs : observations,
                     self.taken_action : actions,
                     self.returns : returns}

        _, loss = session.run([self.train_op, self.loss], feed_dict)

        return float(loss)


class REINFORCE(BaseAgent):
    """
    Monte Carlo implementation of REINFORCE
    No baseline - true Monte Carlo returns used

    args
        env (energy_py environment)
        discount (float)
        brain_path (str) : directory where brain lives
        policy (energy_py policy approximator)
        learning rate (float)

    Monte Carlo REINFORCE is high variance and low bias
    Variance can be reduced through the use of a baseline TODO

    This algorithm requires lots of episodes to run:
    - policy gradient only makes small updates
    - Monte Carlo is high variance (takes a while for expectation to converge)
    - we only update once per episode
    - only learn from samples once

    Reference = Williams (1992)
    """
    def __init__(self,
                 env,
                 discount,
                 policy,
                 lr):

        super().__init__(env, discount)

        #  create the policy function approximator
        self.model_dict = {'input_nodes': self.observation_space.shape[0],
                           'output_nodes': self.action_space.shape[0]*2,
                           'layers': [25, 25],
                           'lr': lr,
                           'action_space': self.action_space}

        self.policy = policy(self.model_dict)

        #  we make a state processor using the observation space
        #  minimums and maximums
        self.state_processor = Standardizer(self.observation_space.shape[0])

        #  we use a normalizer for the returns as well
        #  because we don't want to shift the mean
        self.returns_processor = Normalizer(1)

    def _act(self, **kwargs):
        """
        Act according to the policy network

        args
            observation : np array (1, observation_dim)
            session     : a TensorFlow Session object

        return
            action      : np array (1, num_actions)
        """
        observation = kwargs.pop('observation')
        session = kwargs.pop('session')

        #  scaling the observation for use in the policy network
        scaled_observation = self.state_processor.transform(observation.reshape(1,-1))

        #  generating an action from the policy network
        action, output = self.policy.get_action(session, scaled_observation)

        for i, mean in enumerate(output['means'].flatten()):
            self.memory.info['mean {}'.format(i)].append(mean)

        for i, stdevs in enumerate(output['stdevs'].flatten()):
            self.memory.info['stdevs {}'.format(i)].append(mean)

        logging.debug('scaled_obs {}'.format(scaled_observation))
        logging.debug('action {}'.format(action))

        self.memory.info['scaled_obs'].extend(list(scaled_observation.flatten()))
        self.memory.info['action'].extend(list(action.flatten()))

        logging.debug('means are {}'.format(output['means']))
        logging.debug('stdevs are {}'.format(output['stdevs']))

        return action.reshape(-1, self.action_space.shape[0])

    def _learn(self, **kwargs):
        """
        Update the policy network using the episode experience

        args
            session (object) a TensorFlow Session object
            batch (dict) dictionary of np.arrays

        return
            loss (float)
        """
        batch = kwargs.pop('batch')
        observations = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']

        #  processing our observation
        #  we don't process the action as we take log_prob(action)
        observations = self.state_processor.transform(observations)
        #  we calculcate discountred returns then process
        returns = self.memory.calculate_returns(rewards)
        returns = self.returns_processor.transform(returns)

        logging.debug('observations {}'.format(observations))
        logging.debug('actions {}'.format(actions))
        logging.debug('returns {}'.format(returns))

        loss = self.policy.improve(session,
                                   observations,
                                   actions,
                                   returns)

        self.memory.info['losses'].append(loss)
        logging.info('loss is {:.8f}'.format(loss))

        return loss
