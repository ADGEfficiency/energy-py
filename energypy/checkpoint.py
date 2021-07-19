from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pickle
import tensorflow as tf

import numpy as np

from energypy import json_util, registry, utils
from energypy.agent import memory

from energypy.init import init_nets, init_optimizers, init_writers


def save(
    hyp,
    nets,
    optimizers,
    buffer,
    episode,
    rewards,
    counters,
    paths=None,
    path=None

):
    if paths:
        path = paths['run'] / 'checkpoints' / f'test-episode-{episode}'
    else:
        assert path is not None

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for name, net in nets.items():
        if 'alpha' not in name:
            net.save_weights(path / f'{name}.h5')

    #  save alpha!
    log_alpha = nets['alpha'].numpy()
    np.save(path / 'alpha.npy', log_alpha)

    for name, optimizer in optimizers.items():
        wts = optimizer.get_weights()
        if wts:
            opt_path = path / f'{name}.pkl'
            with opt_path.open('wb') as fi:
                pickle.dump(wts, fi)

    if memory:
        memory.save(buffer, path / 'buffer.pkl')

    if rewards:
        rewards = dict(rewards)
        rewards['time'] = datetime.utcnow().isoformat()
        json_util.save(
            rewards,
            path / 'rewards.json'
        )
    if counters:
        json_util.save(
            dict(counters),
            path / 'counters.json'
        )
    json_util.save(
        hyp,
        path / 'hyperparameters.json'
    )


def get_checkpoint_paths(run):
    checkpoints = Path(run) / 'checkpoints'
    return [p for p in checkpoints.iterdir() if p.is_dir()]


def load(run, full=False):
    """loads all checkpoints, only loading some checkpoint elements"""
    checkpoints = get_checkpoint_paths(run)
    return [load_checkpoint(p, full) for p in checkpoints]


def load_hyp(path):
    return json_util.load(path / 'hyperparameters.json')


def load_checkpoint(path, full=True):
    """full mode loads everything, other mode loads only rewards & counters
    idea is to have a way to quickly evaluate checkpoints without loading what we don't need"""

    path = Path(path)

    hyp = load_hyp(path)

    rewards = json_util.load(path / 'rewards.json')
    rewards.pop('time')
    rewards = defaultdict(list, rewards)
    counters = defaultdict(int, json_util.load(path / 'counters.json'))

    results = {
        'path': path,
        'hyp': hyp,
        'rewards': rewards,
        'counters': counters,
    }

    if full:
        #  catch a wierd error when we load old buffers
        try:
            buffer = memory.load(path / 'buffer.pkl')
        except ModuleNotFoundError:
            print('failed to load buffer due to ModuleNotFoundError')
            buffer = None

        env = registry.make(**hyp['env'])
        nets = init_nets(env, hyp)

        #  awkward
        nets.pop('target_entropy')
        for name, net in nets.items():
            #  awkward
            if 'alpha' not in name:
                net.load_weights(path / f'{name}.h5')
                print(f'loaded {name}')

        log_alpha = nets['alpha']
        saved_log_alpha = np.load(path / 'alpha.npy')
        log_alpha.assign(saved_log_alpha)

        optimizers = init_optimizers(hyp)
        for name, opt in optimizers.items():
            opt_path = path / f'{name}.pkl'

            if opt_path.exists():
                # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
                model = nets[name]
                #  single var
                if 'alpha' in name:
                    wts = [model, ]
                else:
                    wts = model.trainable_variables
                zero_grads = [tf.zeros_like(w) for w in wts]
                opt.apply_gradients(zip(zero_grads, wts))

                with opt_path.open('rb') as fi:
                    opt.set_weights(pickle.load(fi))

        results['env'] = env
        results['nets'] = nets
        results['optimizers'] = optimizers
        results['buffer'] = buffer

    return results


def init_checkpoint(checkpoint_path):
    point = load_checkpoint(checkpoint_path)
    hyp = point['hyp']
    paths = utils.get_paths(hyp)
    counters = point['counters']

    writers = init_writers(counters, paths)

    transition_logger = utils.make_logger('transitions.data', paths['run'])
    c = point

    rewards = point['rewards']
    return {
        'hyp': hyp,
        'paths': paths,
        'counters': counters,
        'env': c['env'],
        'buffer': c['buffer'],
        'nets': c['nets'],
        'writers': writers,
        'optimizers': c['optimizers'],
        'transition_logger': transition_logger,
        'rewards': rewards
    }
