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


def load_episodes(path):
    #  pass in list
    if isinstance(path, list):
        #  of dataframes
        if isinstance(path[0], pd.DataFrame):
            return path
        else:
            #  of paths
            episodes = [Path(p) for p in path]
            print(f'loading {len(episodes)} from list of paths')

    #  pass in directory
    elif Path(path).is_dir() or isinstance(path, str):
        path = Path(path)
        episodes = [p for p in path.iterdir()]
        print(f'loading {len(episodes)} from a directory {path}')
    else:
        path = Path(path)
        assert path.is_file() and path.suffix == '.csv'
        episodes = [path, ]
        print(f'loading from a one file {path}')

    csvs = [pd.read_csv(p, index_col=0) for p in tqdm(episodes) if p.suffix == '.csv']
    parquets = [pd.read_parquet(p) for p in tqdm(episodes) if p.suffix == '.parquet']
    eps = csvs + parquets
    print(f'loaded {len(episodes)}')
    return eps


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


def load_attention_episodes(episodes):
    """
    attention-dataset/train/{features,mask}

    we are given 'attention-dataset/train' -> load features + mask
    """
    #  is this wrong now????
    #  pass in dict
    # if isinstance(episodes, dict):
    #     assert 'features' in episodes.keys()
    #     assert 'mask' in episodes.keys()
    #     assert 'prices' in episodes.keys()
    #     return episodes

    #  pass in list
    #  don't support list of paths - jusht list of dict
    if isinstance(episodes, list):
        return episodes

    #  episodes is a path like 'attention-dataset/train'
    #  automatically find the episode data like features, mask etc

    #  this will find many dates
    path = Path(episodes)
    dates = [p for p in path.iterdir() if p.is_dir()]

    #  list of dicts
    #  eps = [{features: , mask: , prices}]
    eps = []
    for date in dates:
        #  mode = mode of the data
        ep = {}
        for mode in [p for p in date.iterdir() if p.suffix == '.npy']:
            ep[mode.stem] = np.load(mode)

        ep['date'] = date.name
        eps.append(ep)

    return eps


class AbstractDataset(ABC):
    def __init__(self):
        self.resets = {'test': self.reset_test, 'train': self.reset_train}

    def sample_observation(self, cursor):
        """
        returns dict
            {prices: np.array, features: np.array}
        """
        return OrderedDict({k: d[cursor] for k, d in self.episode.items()})

    def reset(self, mode):
        """should return first observation of the current episode"""
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
        self.reset()

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

        train_episodes = load_episodes(train_episodes)
        self.episodes = {
            'train': train_episodes,
            #  random sampling done on train episodes
            'random': train_episodes,
            'test': load_episodes(test_episodes),
        }

        self.episodes['test'] = trim_episodes(self.episodes['test'], self.n_batteries)

        #  test_done is a flag used to control which dataset we sample from
        #  it's a bit hacky
        self.test_done = True
        self.reset()

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
            prices = episode.pop(self.price_col)
            ds['prices'].append(prices.reset_index(drop=True))
            ds['features'].append(episode.reset_index(drop=True))

        self.episode = {
            'prices': pd.concat(ds['prices'], axis=1).values,
            'features': pd.concat(ds['features'], axis=1).values,
        }

        if len(self.test_episodes_queue) == 0:
            self.test_done = True

        return self.sample_observation(0)

    def reset_train(self):
        episodes = random.sample(self.episodes['train'], self.n_batteries)

        ds = defaultdict(list)
        for episode in episodes:
            episode = episode.copy()
            prices = episode.pop(self.price_col)
            ds['prices'].append(prices.reset_index(drop=True).values.reshape(-1, 1, 1))
            ds['features'].append(episode.reset_index(drop=True).values.reshape(prices.shape[0], 1, -1))

        self.episode = {
            'prices': np.concatenate(ds['prices'], axis=1),
            'features': np.concatenate(ds['features'], axis=1),
        }
        return self.sample_observation(0)



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

        train_episodes = load_attention_episodes(train_episodes)

        if test_episodes:
            test_episodes = load_attention_episodes(test_episodes)
        else:
            test_episodes = train_episodes

        #  could be improved, but it's very clear
        # self.episodes = {
        #     'train-features': train_episodes['features'],
        #     'train-mask': train_episodes['mask'],

        #     'random-features': train_episodes['features'],
        #     'random-mask': train_episodes['mask'],

        #     'test-features': trim_episodes(test_episodes['features'], self.n_batteries),
        #     'test-mask': trim_episodes(test_episodes['mask'], self.n_batteries),
        #     'test-prices': trim_episodes(test_episodes['prices'], self.n_batteries)
        # }

        self.episodes = {
            'train': train_episodes,
            'random': train_episodes,
            'test': test_episodes,
        }

        #  start in train mode
        self.test_done = True
        self.reset('train')

    def setup_test(self):
        self.test_done = False
        self.test_episodes_queue = list(range(0, len(self.episodes['test-features'])))
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
                    np.expand_dims(self.episodes[f'test-{var}'][episode_idx], 1)
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
