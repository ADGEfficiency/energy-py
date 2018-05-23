import tensorflow as tf

import energy_py
from energy_py.agents import BaseAgent

from networks import feed_forward_network 
from policies import e_greedy


class DQN(BaseAgent):
    """
    The energy_py implementation of Deep Q-Network

    BaseAgent args (passed as **kwargs)
        sess (tf.Session)
        env (energy_py environment)

    DQN args
        num_discrete_actions (int)
        hiddens (tuple) nodes for each hidden layer of the Q(s,a) approximation

    """

    def __init__(self,
                 total_steps,
                 num_discrete_actions=20,
                 hiddens=(5, 5, 5),
                 initial_epsilon=1.0,
                 final_epsilon=0.05,
                 epsilon_decay_fraction=0.3,
                 **kwargs):

        super().__init__(**kwargs)

        self.discrete_actions = self.env.action_space.discretize(
            num_discrete_actions)
        self.num_actions = self.discrete_actions.shape[1]

        #  not 100% sure about how to do the steps!
        self.step = tf.Variable(0,
            dtype=tf.float32)

        self.online_q_values = feed_forward_network(
            self.observation,
            hiddens,
            self.num_actions,
            'online_q_values',
            output_activation='linear')

        self.epsilon = tf.train.polynomial_decay(
                learning_rate=initial_epsilon,
                global_step=self.learn_step,
                decay_steps=total_steps*epsilon_decay_fraction,
                end_learning_rate=final_epsilon,
                power=1.0,
                name='epsilon')

        #  would also be possible to send the policy object in from the outside
        #  TODO the softmax policyy
        self.policy = e_greedy(self.online_q_values,
                               self.epsilon,
                               self.discrete_actions)


















if __name__ == '__main__':
    env = energy_py.make_env('CartPole')
    with tf.Session() as sess:
        a = DQN(sess=sess, env=env)
