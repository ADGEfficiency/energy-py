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

    q1 = make_qfunc(obs_shape, n_actions, "q1", hyp)
    q1_target = make_qfunc(obs_shape, n_actions, "q1-target", hyp)
    q2 = make_qfunc(obs_shape, n_actions, "q2", hyp)
    q2_target = make_qfunc(obs_shape, n_actions, "q2-target", hyp)

    update_target_network(online=q1, target=q1_target, rho=0.0)
    update_target_network(online=q2, target=q2_target, rho=0.0)
    onlines = [q1, q2]
    targets = [q1_target, q2_target]
    return onlines, targets


def make_qfunc(obs_shape, n_actions, name, hyp):
    """makes a single qfunc"""
    # if not mask_shape:
    #     raise NotImplementedError()

    # in_obs = keras.Input(shape=obs_shape)
    # in_mask = keras.Input(shape=mask_shape)

    #  observation head (with mask) - obs_head is a dense layer
    (in_obs, in_mask), obs_head = energypy.make(
        **hyp["network"], input_shape=obs_shape, outputs=32
    )

    #  action connects into obs_head output, then through dense net to output
    in_act = keras.Input(shape=n_actions)
    _, net = energypy.make(
        name="dense",
        size_scale=hyp["network"]["size_scale"],
        #  these will be concated together
        input_shape=[obs_head, in_act],
        outputs=1,
        neurons=(32, 16),
    )

    # if len(in_obs.shape) == 2:
    #     obs = Flatten()(in_obs)
    #     act = Flatten()(in_act)
    #     inputs = tf.concat([obs, act], axis=1)

    # else:
    #     assert len(in_obs.shape) == 3
    #     act = tf.expand_dims(in_act, 2)
    #     inputs = tf.concat([in_obs, act], axis=1)

    # inp_net, net = energypy.make(**hyp["network"], inputs=inputs, outputs=1)
    # mask = inp_net[1]

    return keras.Model(inputs=[in_obs, in_act, in_mask], outputs=net, name=name)


def update(
    batch, actor, onlines, targets, log_alpha, writer, optimizers, counters, hyp
):
    next_state_act, log_prob, _ = actor(
        (batch["next_observation"], batch["next_observation_mask"])
    )
    next_state_target = minimum_target(
        batch["next_observation"],
        next_state_act,
        batch["next_observation_mask"],
        targets,
    )

    al = tf.exp(log_alpha)
    ga = hyp["gamma"]
    target = batch["reward"] + ga * (1 - batch["done"]) * (
        next_state_target - al * log_prob
    )

    writer.scalar(tf.reduce_mean(target), "qfunc-target", "qfunc-updates")

    for onl, optimizer in zip(onlines, optimizers):
        with tf.GradientTape() as tape:
            q_value = onl(
                [batch["observation"], batch["action"], batch["observation_mask"]]
            )
            loss = tf.keras.losses.MSE(q_value, target)

        grads = tape.gradient(loss, onl.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, onl.trainable_variables))

        writer.scalar(tf.reduce_mean(loss), f"online-{onl.name}-loss", "qfunc-updates")
        writer.scalar(
            tf.reduce_mean(q_value), f"online-{onl.name}-value", "qfunc-updates"
        )

    counters["qfunc-updates"] += 1
