from collections import defaultdict
import tensorflow as tf

from energypy import utils, memory, policy, qfunc, alpha, registry



def init_nets(env, hyp):
    actor = policy.make(env, hyp)
    onlines, targets = qfunc.make(env, size_scale=hyp['size-scale'])
    target_entropy, log_alpha = alpha.make(env, initial_value=hyp['initial-log-alpha'])
    return {
        'actor': actor,
        'online-1': onlines[0],
        'online-2': onlines[1],
        'target-1': targets[0],
        'target-2': targets[1],
        'target_entropy': float(target_entropy),
        'alpha': log_alpha,
    }


def init_writers(counters, paths):
    return {
        'random': utils.Writer('random', counters, paths['run']),
        'test': utils.Writer('test', counters, paths['run']),
        'train': utils.Writer('train', counters, paths['run']),
        'episodes': utils.Writer('episodes', counters, paths['run'])
    }


def init_optimizers(hyp):
    lr = hyp['lr']
    lr_alpha = hyp.get('lr-alpha', lr)

    return {
        'online-1': tf.keras.optimizers.Adam(learning_rate=lr),
        'online-2': tf.keras.optimizers.Adam(learning_rate=lr),
        'actor': tf.keras.optimizers.Adam(learning_rate=lr),
        'alpha': tf.keras.optimizers.Adam(learning_rate=lr_alpha),
    }


def init_fresh(hyp):
    counters = defaultdict(int)
    paths = utils.get_paths(hyp)
    transition_logger = utils.make_logger('transitions.data', paths['run'])

    env = registry.make(**hyp['env'], logger=transition_logger)
    buffer = memory.make(env, hyp)

    nets = init_nets(env, hyp)
    writers = init_writers(counters, paths)
    optimizers = init_optimizers(hyp)

    target_entropy = nets.pop('target_entropy')
    hyp['target-entropy'] = target_entropy

    rewards = defaultdict(list)
    return {
        'hyp': hyp,
        'paths': paths,
        'counters': counters,
        'env': env,
        'buffer': buffer,
        'nets': nets,
        'writers': writers,
        'optimizers': optimizers,
        'transition_logger': transition_logger,
        'rewards': rewards
    }
