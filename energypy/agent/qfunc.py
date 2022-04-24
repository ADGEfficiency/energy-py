from math import prod

import numpy as np
from torch import nn
from torch.distributions import Normal
import torch

import energypy
from energypy.utils import minimum_target
from energypy.agent.target import update_target_network


class DenseQFunc(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        n_outputs: int,
        scale: int = 1,
    ):
        super(DenseQFunc, self).__init__()
        n_inputs = prod(input_shape)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32 * scale),
            nn.ReLU(),
            nn.Linear(32 * scale, 64 * scale),
            nn.ReLU(),
            nn.Linear(64 * scale, n_outputs),
        )

    def forward(self, obs, act):
        obs = torch.from_numpy(obs)
        act = torch.from_numpy(act)
        obs_act = torch.cat(
            [self.flatten(obs), self.flatten(act)],
            axis=1
        )
        dense = self.net(obs_act)
        return dense


def update_target_network(online, target, rho):
    for on, ta in zip(online.parameters(), target.parameters()):
        ta.data.copy_(rho * ta.data + on.data * (1.0 - rho))


def make(env, hyp):
    """makes the two online & two targets"""

    n_actions = sum(env.action_space.shape)

    #  figure out the correct shap
    obs = env.observation_space.sample()
    act = env.action_space.sample()
    obs_act = np.concatenate([obs, act])

    q1 = DenseQFunc(obs_act.shape, n_actions)
    q1_target = DenseQFunc(obs_act.shape, n_actions)

    q2 = DenseQFunc(obs_act.shape, n_actions)
    q2_target = DenseQFunc(obs_act.shape, n_actions)

    update_target_network(online=q1, target=q1_target, rho=0.0)
    update_target_network(online=q2, target=q2_target, rho=0.0)

    onlines = [q1, q2]
    targets = [q1_target, q2_target]
    return onlines, targets


def make_qfunc(obs_shape, n_actions, name, hyp):
    """makes a single qfunc"""

    if hyp['network']['name'] == 'dense':

        #  observation head - obs_head is a dense layer
        in_obs, obs_head = energypy.make(
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
        return keras.Model(inputs=[in_obs, in_act], outputs=net, name=name)

    if hyp['network']['name'] == 'attention':

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
    if hyp['network']['name'] == 'dense':
        next_state_act, log_prob, _ = actor(batch["next_observation"])
        next_state_target = minimum_target(
            (batch["next_observation"],
            next_state_act),
            targets=targets,
        )

    if hyp['network']['name'] == 'attention':
        next_state_act, log_prob, _ = actor(
            (batch["next_observation"], batch["next_observation_mask"])
        )

        next_state_target = minimum_target(
            (batch["next_observation"],
            next_state_act,
            batch["next_observation_mask"]),
            targets=targets,
        )

    al = tf.exp(log_alpha)
    ga = hyp["gamma"]
    target = batch["reward"] + ga * (1 - batch["done"]) * (
        next_state_target - al * log_prob
    )

    writer.scalar(tf.reduce_mean(target), "qfunc-target", "qfunc-updates")

    for onl, optimizer in zip(onlines, optimizers):
        with tf.GradientTape() as tape:
            if hyp['network']['name'] == 'dense':
                q_value = onl(
                    [batch["observation"], batch["action"]]
                )
            if hyp['network']['name'] == 'attention':
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
    return loss
