import numpy as np
import tensorflow as tf
import torch
from torch import nn

import energypy


class SingleParameter(nn.Module):
    def __init__(self):
        super(DensePolicy, self).__init__()
        self.log_alpha = torch.tensor(
            hyp["initial-log-alpha"],
            requires_grad=True,
            dtype=torch.float32
        )

    def forward(self):
        return self.log_alpha


class Alpha(energypy.agent.Base):
    def __init__(self, hyp, device='cpu'):
        self.net = SingleParameter()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=hyp.get('lr-alpha', hyp['lr'])
        )

        target_entropy = -np.product(env.action_space.shape)
        self.target_entropy = torch.tensor(target_entropy)


def make(env, hyp):
    return Alpha(hyp)

# x.backward(torch.tensor(12.4)) - for training later - https://stackoverflow.com/questions/59800247/pytorch-equivalent-of-tf-variable


def update(batch, policy, alpha, hyp, optimizer, counters, writer):

    #  return_numpy as we are not training the policy here
    _, log_prob, _ = policy(
        batch["observation"],
        batch["observation_mask"],
        return_numpy=True
    )
    log_prob = torch.tensor(log_prob)

    loss = -1.0 * torch.mean((torch.exp(alpha.log_alpha) * (log_prob + target_entropy)))

    #  we backproppin' boppin'
    alpha.optimizer.zero_grad()
    loss.backward()
    alpha.optimizer.step()

    writer.scalar(tf.reduce_mean(loss), "alpha-loss", "alpha-updates")
    writer.scalar(tf.exp(log_alpha), "alpha", "alpha-updates")
    writer.scalar(tf.reduce_mean(log_prob), "alpha-log-probs", "alpha-updates")
    counters["alpha-updates"] += 1
