import numpy as np
import tensorflow as tf
import torch


def make(env, initial_value):
    target_entropy = -np.product(env.action_space.shape)
    log_alpha = torch.tensor(
        initial_value,
        requires_grad=True,
        dtype=torch.float32
    )
    return target_entropy, log_alpha

# x.backward(torch.tensor(12.4)) - for training later - https://stackoverflow.com/questions/59800247/pytorch-equivalent-of-tf-variable


def update(batch, policy, log_alpha, hyp, optimizer, counters, writer):

    _, log_prob, _ = policy(batch["observation"], batch["observation_mask"], return_numpy=True)

    target_entropy = torch.tensor(hyp["target-entropy"])
    log_prob = torch.tensor(log_prob)

    loss = -1.0 * torch.mean((torch.exp(log_alpha) * (log_prob + target_entropy)))

    #  we backproppin' boppin'
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.scalar(tf.reduce_mean(loss), "alpha-loss", "alpha-updates")
    writer.scalar(tf.exp(log_alpha), "alpha", "alpha-updates")
    writer.scalar(tf.reduce_mean(log_prob), "alpha-log-probs", "alpha-updates")
    counters["alpha-updates"] += 1
