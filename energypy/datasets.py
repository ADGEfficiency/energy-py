from abc import ABC, abstractmethod

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


def round_nearest(x, divisor):
    return x - (x % divisor)


def trim_episodes(episodes, n_batteries):
#  want test episodes to be a multiple of the number of batteries
    episodes_before = len(episodes)
    lim = round_nearest(len(episodes[:]), n_batteries)
    episodes = episodes[:lim]
    assert len(episodes) % n_batteries == 0
    episodes_after = len(episodes)
    print(f'lost {episodes_before - episodes_after} test episodes due to even multiple')
    return episodes


class AbstractDataset(ABC):
    def __init__(self):
        self.resets = {'test': self.reset_test, 'train': self.reset_train}

    def sample_observation(self, cursor):
        """
        returns dict
            {prices: np.array, features: np.array}
        """
        return OrderedDict({k: d[cursor] for k, d in self.episode.items()})

    def reset(self, mode='train'):
        """returns first observation of the current episode"""
        return self.resets[mode]()

    def setup_test(self):
        """
        called by energypy.main
        not optional - even if dataset doesn't have the concept of test data
        no test data -> setup_test should return True

        maybe could run this when we switch from train to test?
        simpler to be explicit
        """
        return True

    def reset_test(self):
        raise NotImplementedError()

    def reset_train(self):
        raise NotImplementedError()


class RandomDataset(AbstractDataset):
    def __init__(self, n=1000, n_features=3, n_batteries=1, logger=None):
        super(RandomDataset, self).__init__()
        self.dataset = self.make_random_dataset(n, n_features, n_batteries)
        self.test_done = True  #  no notion of test data for random data
        self.n_batteries = n_batteries

    def make_random_dataset(self, n, n_features, n_batteries):
        np.random.seed(42)
        #  (timestep, batteries, features)
        prices = np.random.uniform(0, 100, n*n_batteries).reshape(n, n_batteries, 1)
        features = np.random.uniform(0, 100, n*n_features*n_batteries).reshape(n, n_batteries, n_features)
        return {'prices': prices, 'features': features}


class NEMDataset(AbstractDataset):
    def __init__(
        self,
        n_batteries,
        train_episodes=None,
        test_episodes=None,
        price_col='price [$/MWh]',
        logger=None
    ):
        super(NEMDataset, self).__init__()

        self.n_batteries = n_batteries
        self.price_col = price_col

        train_episodes = self.load_episodes(train_episodes)
        self.episodes = {
            'train': train_episodes,
            #  random sampling done on train episodes
            'random': train_episodes,
            'test': self.load_episodes(test_episodes),
        }

        self.episodes['test'] = trim_episodes(self.episodes['test'], self.n_batteries)

        #  test_done is a flag used to control which dataset we sample from
        #  it's a bit hacky
        self.test_done = True

    def setup_test(self):
        self.test_done = False
        self.test_episodes_queue = list(range(0, len(self.episodes['test'])))
        return self.test_done

    def reset_test(self):

        #  sample the next n_batteries batteries
        episodes = self.test_episodes_queue[:self.n_batteries]

        #  remove the sample from the queue
        self.test_episodes_queue = self.test_episodes_queue[self.n_batteries:]

        #  iterate over episodes to pack them into one many battery episode
        ds = defaultdict(list)
        for episode_idx in episodes:
            episode = self.episodes['test'][episode_idx].copy()
            prices = episode[self.price_col]
            ds['prices'].append(prices.reshape(prices.shape[0], 1, 1))

            features = episode['features']
            ds['features'].append(features.reshape(
                features.shape[0],
                1,
                *features.shape[1:]
            ))

        self.episode = {
            'prices': np.concatenate(ds['prices'], axis=1),
            'features': np.concatenate(ds['features'], axis=1),
        }

        if len(self.test_episodes_queue) == 0:
            self.test_done = True

        return self.sample_observation(0)

    def reset_train(self):
        episodes = random.sample(self.episodes['train'], self.n_batteries)

        #  iterate over episodes to pack them into one many battery episode
        ds = defaultdict(list)
        for episode in episodes:
            prices = episode[self.price_col]
            ds['prices'].append(prices.reshape(prices.shape[0], 1, 1))

            features = episode['features']
            ds['features'].append(features.reshape(
                features.shape[0],
                1,
                *features.shape[1:]
            ))

        self.episode = {
            'prices': np.concatenate(ds['prices'], axis=1),
            'features': np.concatenate(ds['features'], axis=1),
        }
        assert len(self.episode['prices']) == len(self.episode['features'])
        assert self.episode['prices'].ndim == 3
        assert self.episode['features'].ndim == 3

        return self.sample_observation(0)

    def load_episodes(self, episodes):
        #  pass in list of dicts
        #  don't support list of paths - jusht list of dict
        if isinstance(episodes, list):
            if isinstance(episodes[0], dict):
                return episodes

        #  episodes is a path like .data/attention/train

        episodes = Path(episodes)
        out = []
        for ep in [p.name for p in (episodes / 'features').iterdir()]:
            pkg = {}
            for el in ['features', 'prices']:
                pkg[el] = np.load(episodes / el / ep)

            out.append(pkg)
        return out


        # #  pass in list
        # if isinstance(path, list):
        #     #  of dataframes
        #     if isinstance(path[0], pd.DataFrame):
        #         return path
        #     else:
        #         #  of paths
        #         episodes = [Path(p) for p in path]
        #         print(f'loading {len(episodes)} from list of paths')

        # #  pass in directory
        # elif Path(path).is_dir() or isinstance(path, str):
        #     path = Path(path)
        #     episodes = [p for p in path.iterdir()]
        #     print(f'loading {len(episodes)} from a directory {path}')
        # else:
        #     path = Path(path)
        #     assert path.is_file() and path.suffix == '.csv'
        #     episodes = [path, ]
        #     print(f'loading from a one file {path}')

        # csvs = [pd.read_csv(p, index_col=0) for p in tqdm(episodes) if p.suffix == '.csv']
        # parquets = [pd.read_parquet(p) for p in tqdm(episodes) if p.suffix == '.parquet']
        # eps = csvs + parquets
        # print(f'loaded {len(episodes)}')
        # return eps



class NEMDatasetAttention(AbstractDataset):
    """
    features = (batch, n_batteries, sequence_length, n_features)
    mask = (batch, n_batteries, sequence_length, sequence_length)
    prices = (batch, 1)
    """
    def __init__(
        self,
        n_batteries,
        train_episodes=None,
        test_episodes=None,
        price_col='price [$/MWh]',
        logger=None
    ):
        super(NEMDatasetAttention, self).__init__()

        self.n_batteries = n_batteries
        self.price_col = price_col

        train_episodes = self.load_episodes(train_episodes)

        if test_episodes:
            test_episodes = self.load_episodes(test_episodes)
        else:
            test_episodes = train_episodes

        self.episodes = {
            'train': train_episodes,
            'random': train_episodes,
            'test': test_episodes,
        }

        #  start in train mode
        self.test_done = True

    def setup_test(self):
        self.test_done = False
        self.test_episodes_queue = list(range(0, len(self.episodes['test'])))
        return self.test_done

    def reset_test(self):
        #  sample the next n_batteries episodes
        episodes_idxs = self.test_episodes_queue[:self.n_batteries]

        #  remove the sample from the queue
        self.test_episodes_queue = self.test_episodes_queue[self.n_batteries:]

        #  iterate over episodes to pack them into one many battery episode
        ds = defaultdict(list)
        for episode_idx in episodes_idxs:
            for var in ['features', 'mask', 'prices']:
                ds[var].append(
                    np.expand_dims(self.episodes[f'test'][episode_idx][var], 1)
                )

        self.episode = {}
        for var in ['features', 'mask', 'prices']:
            self.episode[var] = np.concatenate(ds[var], axis=1)

        self.test_done = len(self.test_episodes_queue) == 0
        return self.sample_observation(0)

    def reset_train(self):
        #  sample the next n_batteries episodes
        episodes_idxs = random.sample(range(len(self.episodes['train'])), self.n_batteries)

        #  iterate over episodes to pack them into one many battery episode
        ds = defaultdict(list)
        for episode_idx in episodes_idxs:
            episode = self.episodes['train'][episode_idx]
            for var in ['features', 'mask', 'prices']:
                ds[var].append(np.expand_dims(episode[var], 1))

        self.episode = {}
        for var in ['features', 'mask', 'prices']:
            self.episode[var] = np.concatenate(ds[var], axis=1)

        return self.sample_observation(0)

    def load_episodes(self, episodes):
        """
        ./data/attention/{test,train}/{features,prices,mask}/%Y-%m-%d.npy
        """
        #  pass in list of dicts
        #  don't support list of paths - jusht list of dict
        if isinstance(episodes, list):
            if isinstance(episodes[0], dict):
                return episodes

        #  episodes is a path like .data/attention/train

        episodes = Path(episodes)
        out = []
        for ep in [p.name for p in (episodes / 'features').iterdir()]:
            pkg = {}
            for el in ['features', 'mask', 'prices']:
                pkg[el] = np.load(episodes / el / ep)

            out.append(pkg)
        return out


