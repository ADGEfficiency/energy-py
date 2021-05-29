from collections import OrderedDict, defaultdict
import json

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def make_perfect_forecast(prices, horizon):
    prices = np.array(prices).reshape(-1, 1)
    forecast = np.hstack([np.roll(prices, -i) for i in range(0, horizon)])
    return forecast[:-(horizon-1), :]


def load_episodes(path):
    #  pass in list of filepaths
    if isinstance(path, list):
        if isinstance(path[0], pd.DataFrame):
            #  list of dataframes?
            return path
        else:
            #  list of paths
            episodes = path
            print(f'loading {len(episodes)} from list')
            eps = [pd.read_csv(p, index_col=0) for p in tqdm(episodes)]
            print(f'loaded {len(episodes)} from list')
            return eps

    #  pass in directory
    elif Path(path).is_dir() or isinstance(path, str):
        path = Path(path)
        episodes = [p for p in path.iterdir() if p.suffix == '.csv']
    else:
        path = Path(path)
        assert path.is_file() and path.suffix == '.csv'
        episodes = [path, ]

    print(f'loading {len(episodes)} from {path.name}')
    eps = [pd.read_csv(p, index_col=0) for p in tqdm(episodes)]
    print(f'loaded {len(episodes)} from {path.name}')
    return eps


def round_nearest(x, divisor):
    return x - (x % divisor)


class AbstractDataset:
    def reset(self, mode=None):
        raise NotImplementedError()

    def setup_test(self):
        raise NotImplementedError()

    def get_data(self, cursor):
        return OrderedDict({k: d[cursor] for k, d in self.dataset.items()})


class RandomDataset(AbstractDataset):
    def __init__(self, n=1000, n_features=3, n_batteries=1, logger=None):
        self.dataset = self.make_random_dataset(n, n_features, n_batteries)
        self.test_done = True
        self.reset()

    def reset(self, mode=None):
        pass

    def setup_test(self):
        pass

    def make_random_dataset(self, n, n_features, n_batteries):
        np.random.seed(42)
        #  (timestep, batteries, features)
        prices = np.random.uniform(0, 100, n*n_batteries).reshape(n, n_batteries, 1)
        features = np.random.uniform(0, 100, n*n_features*n_batteries).reshape(n, n_batteries, n_features)
        return {'prices': prices, 'features': features}


class NEMDataset(AbstractDataset):
    def __init__(self, n_batteries, train_episodes=None, test_episodes=None, logger=None):
        self.n_batteries = n_batteries

        #  TODO - use of the random key here is awkard
        #  using it to get the random policy sampling to play nice

        train_episodes = load_episodes(train_episodes)
        self.episodes = {
            'train': train_episodes,
            'random': train_episodes,
            'test': load_episodes(test_episodes),
        }

        #  want test episodes to be a multiple of the number of batteries
        episodes_before = len(self.episodes['test'])
        lim = round_nearest(len(self.episodes['test'][:]), self.n_batteries)
        self.episodes['test'] = self.episodes['test'][:lim]
        assert len(self.episodes['test']) % self.n_batteries == 0
        episodes_after = len(self.episodes['test'])
        print(f'lost {episodes_before - episodes_after} test episodes due to even multiple')

        self.test_done = True
        self.reset()

    def reset(self, mode='train'):
        if mode == 'test':
            return self.reset_test()
        else:
            return self.reset_train()

    def reset_train(self):
        episodes = random.sample(self.episodes['train'], self.n_batteries)

        ds = defaultdict(list)
        for episode in episodes:
            episode = episode.copy()
            prices = episode.pop('price [$/MWh]')
            ds['prices'].append(prices.reset_index(drop=True).values.reshape(-1, 1, 1))
            ds['features'].append(episode.reset_index(drop=True).values.reshape(prices.shape[0], 1, -1))

        #  TODO could call this episode
        self.dataset = {
            'prices': np.concatenate(ds['prices'], axis=1),
            'features': np.concatenate(ds['features'], axis=1),
        }
        return self.get_data(0)

    def setup_test(self):
        #  called during main?
        self.test_done = False
        self.test_episodes_idx = list(range(0, len(self.episodes['test'])))
        return self.test_done

    def reset_test(self):
        episodes = self.test_episodes_idx[:self.n_batteries]
        self.test_episodes_idx = self.test_episodes_idx[self.n_batteries:]

        ds = defaultdict(list)
        for episode in episodes:
            episode = self.episodes['test'][episode].copy()
            prices = episode.pop('price [$/MWh]')
            ds['prices'].append(prices.reset_index(drop=True))
            ds['features'].append(episode.reset_index(drop=True))

        #  TODO could call this episode
        self.dataset = {
            'prices': pd.concat(ds['prices'], axis=1).values,
            'features': pd.concat(ds['features'], axis=1).values,
        }

        if len(self.test_episodes_idx) == 0:
            self.test_done = True

        return self.get_data(0)
