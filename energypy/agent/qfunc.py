import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten

import energypy
from energypy.agent.target import update_target_network
from energypy.networks import dense, attention
from energypy.utils import minimum_target


def make(env, hyp):
    """makes the two online & two targets"""
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.shape

    q1 = make_qfunc(obs_shape, n_actions, 'q1', size_scale)
    q1_target = make_qfunc(obs_shape, n_actions, 'q1-target', size_scale)
    q2 = make_qfunc(obs_shape, n_actions, 'q2', size_scale)
    q2_target = make_qfunc(obs_shape, n_actions, 'q2-target', size_scale)

    update_target_network(online=q1, target=q1_target, rho=0.0)
    update_target_network(online=q2, target=q2_target, rho=0.0)
    onlines = [q1, q2]
    targets = [q1_target, q2_target]
    return onlines, targets


def make_qfunc(obs_shape, n_actions, name, hyp):
    """makes a single qfunc"""

    in_obs = keras.Input(shape=obs_shape)
    in_act = keras.Input(shape=n_actions)

    obs = Flatten()(in_obs)
    act = Flatten()(in_act)

    inputs = tf.concat([obs, act], axis=1)
    _, net = energypy.make(**hyp['network'], inputs=inputs, outputs=1)

    return keras.Model(
        inputs=[in_obs, in_act],
        outputs=net,
        name=name
    )


def update(
    batch,
    actor,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizers,
    counters,
    hyp
):
    next_state_act, log_prob, _ = actor(batch['next_observation'])
    next_state_target = minimum_target(batch['next_observation'], next_state_act, targets)

    al = tf.exp(log_alpha)
    ga = hyp['gamma']
    target = batch['reward'] + ga * (1 - batch['done']) * (next_state_target - al * log_prob)

    writer.scalar(
        tf.reduce_mean(target),
        'qfunc-target',
        'qfunc-updates'
    )

    for onl, optimizer in zip(onlines, optimizers):
        with tf.GradientTape() as tape:
            q_value = onl([batch['observation'], batch['action']])
            loss = tf.keras.losses.MSE(q_value, target)

        grads = tape.gradient(loss, onl.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, onl.trainable_variables))

        writer.scalar(
            tf.reduce_mean(loss),
            f'online-{onl.name}-loss',
            'qfunc-updates'
        )
        writer.scalar(
            tf.reduce_mean(q_value),
            f'online-{onl.name}-value',
            'qfunc-updates'
        )

    counters['qfunc-updates'] += 1
