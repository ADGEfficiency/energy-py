import numpy as np
import tensorflow as tf

import energy_py

from new_dqn import DQN

env = energy_py.make_env('Battery')
obs = env.observation_space.sample()
discount = 0.95

samples = 10

rewards = np.random.uniform(0, 10, samples).reshape(-1)

terms = np.random.randint(0, 2, 10).astype(np.bool)

next_obs = np.array(
    [env.observation_space.sample() for _ in range(samples)]
).reshape(samples, *env.obs_space_shape) 

def test_dqn():
    tf.reset_default_graph()
    with tf.Session() as sess:
        a = DQN(sess=sess, env=env, total_steps=10,
                discount=discount)

        sess.run(tf.global_variables_initializer())
        #  test of the q_selected action part of the graph
        q_vals = sess.run(
            a.online_q_values,
            {a.observation: obs}
        )

        selected = np.random.randint(low=0,
                                     high=a.num_actions,
                                     size=20).reshape(-1)
        q_selected = sess.run(
            a.q_selected_actions,
            {a.observation: obs,
             a.selected_actions: selected},
        )

        q_check = q_vals.reshape(-1)[selected]

        #  our first test of dqn
        assert q_check.all() == q_selected.all()

        action = a.act(obs)

        #  below I check the terminal are masking the target network correctly
        unmaksed_next_q = sess.run(
            a.target_q_values,
            {a.next_observation: next_obs}
        )

        masked_next_q = sess.run(
            a.next_state_max_q,
            {a.next_observation: next_obs,
             a.terminal: terms}
        )

        #  this check is mirroring the tensorflow code exactly :D
        check_masked_next_q = np.where(terms.reshape(-1, 1),
                                       unmaksed_next_q,
                                       np.zeros_like(unmaksed_next_q))

        assert check_masked_next_q.all() == masked_next_q.all()

        #  checking normal q-learning bellman target

        bellman = sess.run(
            a.bellman,
            {a.next_observation: next_obs,
             a.terminal: terms,
             a.reward: rewards}
        )

        bellman_check = rewards + discount * np.max(masked_next_q, axis=1)
        assert bellman_check.all() == bellman.all()


def test_ddqn():
    tf.reset_default_graph()
    with tf.Session() as sess:
        a = DQN(sess=sess, env=env, total_steps=10,
                discount=discount, double_q=True)

        sess.run(tf.global_variables_initializer())
        #  test that our weights are being shared correctly
        online_vals = sess.run(
            a.online_q_values,
            {a.observation: next_obs}
        )

        online_copy_vals = sess.run(
            a.online_next_obs_q,
            {a.next_observation: next_obs}
        )
        assert online_copy_vals.all() == online_vals.all()

        #  lets check the double q target creation

        #  get the optimal next action suggested by the online net
        online_next_obs_q = sess.run(
            a.online_next_obs_q,
            {a.next_observation: next_obs}
        )

        online_acts = np.argmax(online_next_obs_q, axis=1)

        #  use this as an integer index on the target net Q(s,a) approximation

        target_net_q_values = sess.run(
            a.target_q_values,
            {a.next_observation: next_obs}
        )

        selected_target_net_q = target_net_q_values[np.arange(samples),
                                                    online_acts]

        masked = np.where(terms, 
                          selected_target_net_q, 
                          np.zeros(samples))

        bellman_check = rewards + discount * masked

        bell = sess.run(
            a.bellman,
            {a.next_observation: next_obs,
             a.terminal: terms,
             a.reward: rewards}
        )

        assert bellman_check.all() == bell.all()
