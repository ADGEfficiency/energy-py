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

    def forward(self, x, mask=None):
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


class Actor(energypy.agent.Base):
    def __init__(
        self,
        input_shape: tuple,
        n_outputs: int,
        scale: int = 1,
        device='cpu'
    ):
        self.net = DensePolicy(input_shape, n_outputs, scale).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()


def make(env, hyp, device='cpu'):
    """makes an Actor - always with DensePolicy"""
    obs_shape = env.elements_helper['observation']
    n_actions = np.zeros(env.elements_helper['action']).size
    return Actor(
        input_shape=obs_shape,
        n_outputs=n_actions * 2,
        device=device,
        scale=hyp['network']['scale']
    )


def update(
    batch,
    policy,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizer,
    counters,
    hyp
):
    al = torch.exp(torch.tensor(log_alpha(return_numpy=True)))

    state_act, log_prob, _ = policy(
        batch["observation"], batch["observation_mask"], return_numpy=False
    )
    policy_target = minimum_target(
        (batch["observation"], state_act, batch["observation_mask"]), targets
    )
    loss = torch.mean(al * log_prob - policy_target)

    #  we backproppin' boppin'
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()

    #  should handle this in writer.scalar method
    writer.scalar(torch.mean(policy_target).detach().numpy(), "policy-target", "policy-updates")
    writer.scalar(loss.detach().numpy(), "policy-loss", "policy-updates")
    writer.scalar(torch.mean(log_prob).detach().numpy(), "policy-log-prob", "policy-updates")

    counters["policy-updates"] += 1
    return loss
