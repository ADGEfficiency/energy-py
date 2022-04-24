from math import prod

import numpy as np
from torch import nn
from torch.distributions import Normal
import torch

import energypy
from energypy.utils import minimum_target


#  clip as per stable baselines
log_stdev_low, log_stdev_high = -20, 2
epsilon = 1e-6


class DensePolicy(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        n_outputs: int,
        scale: int = 1,
    ):
        super(DensePolicy, self).__init__()
        n_inputs = prod(input_shape)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32 * scale),
            nn.ReLU(),
            nn.Linear(32 * scale, 64 * scale),
            nn.ReLU(),
            nn.Linear(64 * scale, n_outputs),
        )

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.flatten(x)
        dense = self.net(x)
        mean, log_stdev = torch.split(dense, 1, 1)
        log_stdev = torch.clamp(log_stdev, min=log_stdev_low, max=log_stdev_high)
        stdev = torch.exp(log_stdev)

        normal = Normal(mean, stdev)

        #  unsquashed
        action = normal.sample()
        log_prob = normal.log_prob(action)

        #  squashed
        action = torch.tanh(action)
        deterministic_action = torch.tanh(mean)

        log_prob = torch.sum(
            log_prob - torch.log(1 - action ** 2 + epsilon), dim=1, keepdim=True
        )
        return action, log_prob, deterministic_action



def make(env, hyp, device='cpu'):
    """makes a Dense policy - always"""
    obs_shape = env.observation_space.shape
    n_actions = np.zeros(env.action_space.shape).size
    return DensePolicy(
        input_shape=obs_shape,
        n_outputs=n_actions * 2
    ).to(device)


def update(
    batch,
    actor,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizer,
    counters,
    hyp
):
    al = tf.exp(log_alpha)
    with tf.GradientTape() as tape:

        if hyp['network']['name'] == 'dense':
            state_act, log_prob, _ = actor(batch["observation"])
            policy_target = minimum_target(
                (batch["observation"], state_act), targets
            )
        if hyp['network']['name'] == 'attention':
            state_act, log_prob, _ = actor(
                (batch["observation"], batch["observation_mask"])
            )
            policy_target = minimum_target(
                (batch["observation"], state_act, batch["observation_mask"]), targets
            )

        loss = tf.reduce_mean(al * log_prob - policy_target)

    grads = tape.gradient(loss, actor.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    writer.scalar(tf.reduce_mean(policy_target), "policy-target", "policy-updates")
    writer.scalar(tf.reduce_mean(loss), "policy-loss", "policy-updates")
    writer.scalar(tf.reduce_mean(log_prob), "policy-log-prob", "policy-updates")

    counters["policy-updates"] += 1
    return loss


if __name__ == '__main__':
    #  here just experimenting
    x = env.observation_space.sample().reshape(1, -1)
    x = torch.from_numpy(x).to('cpu')
    net.forward(x)
