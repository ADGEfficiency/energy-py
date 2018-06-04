import numpy as np
import tensorflow as tf

from energy_py.common.policies import epsilon_greedy_policy


def test_e_greedy_policy():
    #  setup for testing
    num_samples = 5
    num_actions = 3
    act_dims = 4

    test_q_values = np.zeros(num_samples * num_actions).reshape(num_samples, -1)
    test_q_values[0, 1] = 1
    test_q_values[1, 2] = 1
    test_q_values[2, 0] = 1

    discrete_actions = np.array(
        [np.random.uniform(size=act_dims)
         for _ in range(num_samples)]).reshape(num_samples, -1)

    #  placeholders for testing
    q_values = tf.placeholder(shape=(None, num_actions), dtype=tf.float32)
    epsilon = tf.placeholder(shape=(), dtype=tf.float32)

    #  construct the tf graph for testing
    step = tf.placeholder(shape=(), name='learning_step', dtype=tf.int64)

    decay_steps = 10
    epsilon, policy = epsilon_greedy_policy(
        q_values,
        discrete_actions,
        step,
        decay_steps=decay_steps,
        initial_epsilon=1.0,
        final_epsilon=0.0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #  check that epsilon at zero gives us the best actions
        optimals = sess.run(
            policy,
            {q_values: test_q_values,
             step: decay_steps + 1}
        )

        np.testing.assert_array_equal(optimals[0], discrete_actions[1])
        np.testing.assert_array_equal(optimals[1], discrete_actions[2])
        np.testing.assert_array_equal(optimals[2], discrete_actions[0])

        #  check that epislon at one gives random actions
        randoms = sess.run(policy,
                     {q_values: test_q_values,
                      step: 0})

        one_different = False

        for opt, random in zip (optimals, randoms):
            if np.array_equal(opt, random):
                pass
            else:
                one_different = True

        assert one_different


if __name__ == '__main__':
    test_e_greedy_policy()
