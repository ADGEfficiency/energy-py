import numpy as np
import tensorflow as tf
import torch


# def make(env, initial_value):
#     target_entropy = -np.product(env.action_space.shape)

#     log_alpha = tf.Variable(
#         initial_value, trainable=True, name="log-alpha", dtype="float32"
#     )
#     return target_entropy, log_alpha

def make(env, initial_value):
    target_entropy = -np.product(env.action_space.shape)
    log_alpha = torch.tensor(
        initial_value,
        requires_grad=True,
        dtype=torch.float32
    )
    return target_entropy, log_alpha

# x.backward(torch.tensor(12.4)) - for training later - https://stackoverflow.com/questions/59800247/pytorch-equivalent-of-tf-variable


def update(batch, actor, log_alpha, hyp, optimizer, counters, writer):
    target_entropy = hyp["target-entropy"]

    if hyp['network']['name'] == 'dense':
        _, log_prob, _ = actor((batch["observation"]))
    if hyp['network']['name'] == 'attention':
        _, log_prob, _ = actor((batch["observation"], batch["observation_mask"]))

    with tf.GradientTape() as tape:
        loss = -1.0 * tf.reduce_mean((tf.exp(log_alpha) * (log_prob + target_entropy)))

    grad = tape.gradient(loss, log_alpha)
    optimizer.apply_gradients(
        zip(
            [
                grad,
            ],
            [
                log_alpha,
            ],
        )
    )

    writer.scalar(tf.reduce_mean(loss), "alpha-loss", "alpha-updates")
    writer.scalar(tf.exp(log_alpha), "alpha", "alpha-updates")
    writer.scalar(tf.reduce_mean(log_prob), "alpha-log-probs", "alpha-updates")
    counters["alpha-updates"] += 1
