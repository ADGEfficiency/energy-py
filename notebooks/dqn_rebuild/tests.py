import numpy as np
import tensorflow as tf

import energy_py

from new_dqn import DQN
from networks import make_copy_ops, make_vars

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
             a.selected_action_indicies: selected},
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

        #  test that our weights are being shared correctly
        online_vals = sess.run(
            a.online_q_values,
            {a.observation: next_obs}
        )

        online_copy_vals = sess.run(
            a.online_next_obs_q,
            {a.next_observation: next_obs}
        )
        print(online_vals[:5])
        print(online_copy_vals[:5])
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


def test_copy_ops():
    tf.reset_default_graph()

    with tf.Session() as sess:

        with tf.variable_scope('online'):
            online_params = make_vars(4)

        with tf.variable_scope('target'):
            target_params = make_vars(4)

        copy_ops, tau = make_copy_ops(online_params, target_params)
        sess.run(tf.global_variables_initializer())

        online_vals, target_vals = sess.run(
            [online_params, target_params]
        )

        assert np.sum(online_vals) != np.sum(target_vals)

        _  = sess.run(copy_ops, {tau: 1.0})

        online_vals, target_vals = sess.run(
            [online_params, target_params]
        )
        assert np.sum(online_vals) == np.sum(target_vals)

def test_target_net_weight_init():
    tf.reset_default_graph()
    with tf.Session() as sess:
        a = DQN(sess=sess, env=env, total_steps=10,
                discount=discount)

        online_vals, target_vals = sess.run(
            [a.online_q_values, a.target_q_values],
            {a.observation: obs,
             a.next_observation: obs}
        )

        #  equal because we intialize target net weights in the init of DQN
        assert np.sum(online_vals) == np.sum(target_vals)


def test_train_op():
    tf.reset_default_graph()
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

        obs = obs.reshape(1, -1)
        #  we haven't learnt yet so our online & target networks should
        #  be the same.  check by comparing the sums of the outputs of each
        online_vals, target_vals = sess.run(
            [agent.online_q_values, agent.target_q_values],
            {agent.observation: obs,
             agent.next_observation: obs}
        )
        assert np.sum(online_vals) == np.sum(target_vals)

        #  learn - changing the online network weights
        agent.learn()

        online_vals, target_vals = sess.run(
            [agent.online_q_values, agent.target_q_values],
            {agent.observation: obs,
             agent.next_observation: obs}
        )
        assert np.sum(online_vals) != np.sum(target_vals)

        #  using copy ops to make online & target the same again
        _ = sess.run(
            agent.copy_ops,
            {agent.tau: 1}
        )

        online_vals, target_vals = sess.run(
            [agent.online_q_values, agent.target_q_values],
            {agent.observation: obs,
             agent.next_observation: obs}
        )
        assert np.sum(online_vals) == np.sum(target_vals)


discrete_actions = np.array([0.0 , 0.0,
                             0.0 , 0.5,
                             0.0, 1.0,
                             0.0, 1.5,
                             0.0, 2.0]).reshape(5, 2)
test_sub_arrays = [
    ([0.0, 2.0], 4),
    ([0.0, 1.0], 2),
    ([0.0, 0.0], 0),
]

from utils import find_sub_array_in_2D_array


def np_test_find_action_in_discrete_actions():
    for sub_array, true_index in test_sub_arrays:

        sub_array = np.array(sub_array).reshape(2)

        assert find_sub_array_in_2D_array(
            sub_array, discrete_actions) == true_index


