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

    def forward(self, obs, act, mask=None):

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)

        if isinstance(act, np.ndarray):
            act = torch.from_numpy(act)

        obs_act = torch.cat(
            [self.flatten(obs), self.flatten(act)],
            axis=1
        )
        dense = self.net(obs_act)
        return dense


def update_target_network(online, target, rho):
    for on, ta in zip(online.net.parameters(), target.net.parameters()):
        ta.data.copy_(rho * ta.data + on.data * (1.0 - rho))


class QFunc(energypy.agent.Base):
    def __init__(
        self,
        name,
        input_shape: tuple,
        n_outputs: int,
        scale: int = 1,
        device='cpu',
    ):
        self.name = name
        self.net = DenseQFunc(input_shape, n_outputs, scale).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()


def make(env, hyp):
    """makes the two online & two targets"""
    n_actions = sum(env.action_space.shape)

    scale = hyp['network']['scale']

    #  figure out the correct shape
    # obs = env.observation_space.sample()
    # act = env.action_space.sample()
    obs_shape = env.elements_helper['observation']
    n_actions = env.elements_helper['action']

    obs_act_shape = np.array(obs_shape) + np.array(n_actions)
    obs_act_shape = tuple(obs_act_shape)

    q1 = QFunc('online-1', obs_act_shape, *n_actions, scale=scale)
    q1_target = QFunc('target-1', obs_act_shape, *n_actions, scale=scale)

    q2 = QFunc('online-2', obs_act_shape, *n_actions, scale=scale)
    q2_target = QFunc('target-2', obs_act_shape, *n_actions, scale=scale)

    update_target_network(online=q1, target=q1_target, rho=0.0)
    update_target_network(online=q2, target=q2_target, rho=0.0)

    onlines = [q1, q2]
    targets = [q1_target, q2_target]
    return onlines, targets


# def make_qfunc(obs_shape, n_actions, name, hyp):
#     """makes a single qfunc"""

#     if hyp['network']['name'] == 'dense':

#         #  observation head - obs_head is a dense layer
#         in_obs, obs_head = energypy.make(
#             **hyp["network"], input_shape=obs_shape, outputs=32
#         )

#         #  action connects into obs_head output, then through dense net to output
#         in_act = keras.Input(shape=n_actions)
#         _, net = energypy.make(
#             name="dense",
#             size_scale=hyp["network"]["size_scale"],
#             #  these will be concated together
#             input_shape=[obs_head, in_act],
#             outputs=1,
#             neurons=(32, 16),
#         )
#         return keras.Model(inputs=[in_obs, in_act], outputs=net, name=name)

#     if hyp['network']['name'] == 'attention':

#         #  observation head (with mask) - obs_head is a dense layer
#         (in_obs, in_mask), obs_head = energypy.make(
#             **hyp["network"], input_shape=obs_shape, outputs=32
#         )

#         #  action connects into obs_head output, then through dense net to output
#         in_act = keras.Input(shape=n_actions)
#         _, net = energypy.make(
#             name="dense",
#             size_scale=hyp["network"]["size_scale"],
#             #  these will be concated together
#             input_shape=[obs_head, in_act],
#             outputs=1,
#             neurons=(32, 16),
#         )

#         # if len(in_obs.shape) == 2:
#         #     obs = Flatten()(in_obs)
#         #     act = Flatten()(in_act)
#         #     inputs = tf.concat([obs, act], axis=1)

#         # else:
#         #     assert len(in_obs.shape) == 3
#         #     act = tf.expand_dims(in_act, 2)
#         #     inputs = tf.concat([in_obs, act], axis=1)

#         # inp_net, net = energypy.make(**hyp["network"], inputs=inputs, outputs=1)
#         # mask = inp_net[1]

#         return keras.Model(inputs=[in_obs, in_act, in_mask], outputs=net, name=name)


def update(
    batch, policy, onlines, targets, log_alpha, writer, optimizers, counters, hyp
):
    next_state_act, log_prob, _ = policy(
        batch["next_observation"], batch["next_observation_mask"], return_numpy=True
    )
    next_state_act = torch.tensor(next_state_act)
    log_prob = torch.tensor(log_prob)

    next_state_target = minimum_target(
        (batch["next_observation"],
        next_state_act,
        batch["next_observation_mask"]),
        targets=targets,
    )

    al = torch.exp(torch.tensor(log_alpha(return_numpy=True)))
    ga = hyp["gamma"]

    batch['reward'] = torch.from_numpy(batch['reward'])
    batch['done'] = torch.from_numpy(batch['done'].astype(np.int32))

    #  all torch tensors
    target = batch["reward"] + ga * (1 - batch["done"]) * (
        next_state_target - al * log_prob
    )
    writer.scalar(torch.mean(target).detach().numpy(), "qfunc-target", "qfunc-updates")

    total_loss = 0
    for onl in onlines:
        onl.optimizer.zero_grad()

        q_value = onl(
            batch["observation"], batch["action"], batch["observation_mask"], return_numpy=False
        )
        loss = onl.loss(q_value, target)
        total_loss += loss

        # where to clip gradients TODO ???
        writer.scalar(loss.detach().numpy(), f"online-{onl.name}-loss", "qfunc-updates")
        writer.scalar(
            q_value.detach().numpy().mean(), f"online-{onl.name}-value", "qfunc-updates"
        )

    #  https://www.pythonfixing.com/2021/12/fixed-calling-backward-function-for-two.html

    #  i have NO IDEA if this is correct
    #  the summing of losses together is a bit :/
    #  I'm hoping that pytorch is clever enough to only put each portion of the loss onto
    #  the correct qfunc!

    #  I think summing losses = wrong
    #  suggest writing two funcs and calling separately....

    total_loss.backward()
    [onl.optimizer.step() for onl in onlines]

    counters["qfunc-updates"] += 1

    return total_loss
