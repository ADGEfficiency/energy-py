from collections import defaultdict
import tensorflow as tf

from energypy import utils, memory, actor, qfunc, alpha, registry
import torch


def init_nets(env, hyp):
    act = actor.make(env, hyp)

    #  turn into test
    # x = torch.from_numpy(env.observation_space.sample().reshape(1, -1))
    # act(x)

    onlines, targets = qfunc.make(env, hyp)
    #  turn into test
    # obs = torch.from_numpy(env.observation_space.sample().reshape(1, -1))
    # act = torch.from_numpy(env.action_space.sample().reshape(1, -1))
    # onlines[0](obs, act)

    al = alpha.make(env, hyp)
    return {
        "actor": act,
        "online-1": onlines[0],
        "online-2": onlines[1],
        "target-1": targets[0],
        "target-2": targets[1],
        "target_entropy": float(al.target_entropy.detach().numpy()),
        "alpha": al,
    }


def init_writers(counters, paths):
    return {
        "random": utils.Writer("random", counters, paths["run"]),
        "test": utils.Writer("test", counters, paths["run"]),
        "train": utils.Writer("train", counters, paths["run"]),
        "episodes": utils.Writer("episodes", counters, paths["run"]),
    }


def init_optimizers(hyp):
    lr = hyp["lr"]
    # lr_alpha = hyp.get("lr-alpha", lr)

    return {
        "online-1": tf.keras.optimizers.Adam(learning_rate=lr),
        "online-2": tf.keras.optimizers.Adam(learning_rate=lr),
        "actor": tf.keras.optimizers.Adam(learning_rate=lr),
        "alpha": None,
    }


def init_fresh(hyp):
    counters = defaultdict(int)
    paths = utils.get_paths(hyp)

    env = registry.make(**hyp["env"])
    buffer = memory.make(env, hyp)

    nets = init_nets(env, hyp)
    writers = init_writers(counters, paths)
    optimizers = init_optimizers(hyp)

    target_entropy = nets.pop("target_entropy")
    hyp["target-entropy"] = target_entropy

    rewards = defaultdict(list)
    return {
        "hyp": hyp,
        "paths": paths,
        "counters": counters,
        "env": env,
        "buffer": buffer,
        "nets": nets,
        "writers": writers,
        "optimizers": optimizers,
        "rewards": rewards,
    }
