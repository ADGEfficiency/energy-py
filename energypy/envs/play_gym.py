import argparse
from collections import defaultdict
import json
from pathlib import Path
import imageio

from energypy import checkpoint, policy, utils, json_util
from energypy.utils import get_paths, get_latest_run
from energypy import utils, registry

import numpy as np


def get_best_checkpoint(checkpoints):
    n_test_episodes = checkpoints[0]['hyp']['n-tests']
    rews = [np.mean(c['rewards']['test-reward'][-n_test_episodes:]) for c in checkpoints]
    best = np.argmax(rews)
    best = checkpoints[best]
    path = best['path']
    print(f'\nfound best checkpoint at {path}')
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    run_path = args.run

    checkpoints = checkpoint.load(run_path)
    checkpoint = get_best_checkpoint(checkpoints)
    print(checkpoint.keys())

    hyp = checkpoint['hyp']
    env = registry.make('lunar')

    actor = checkpoint['nets']['actor']

    env.reset()
    obs = env.reset().reshape(1, -1)
    done = False
    episode_reward = 0

    frames = []
    while not done:
        _, _, action = actor(obs)
        frames.append(env.env.render('rgb_array'))
        next_obs, reward, done = env.step(np.array(action))
        episode_reward += reward
        obs = next_obs

    print(episode_reward)

    print('saving gif')
    imageio.mimsave('./render.gif', frames, fps=44)
