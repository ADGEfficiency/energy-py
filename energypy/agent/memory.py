from pathlib import Path
import pickle

import numpy as np


def make(env, hyp):
    buffer_path = hyp.get('buffer')

    if (buffer_path == 'new') or (buffer_path is None):
        print(f' init buffer')
        return Buffer(env.elements, size=hyp['buffer-size'])

    try:
        buffer = load(buffer_path)
        assert buffer.full
        return buffer

    except FileNotFoundError:
        print(f' failed to load {buffer_path}, init buffer')
        return Buffer(env.elements, size=hyp['buffer-size'])


def save(buffer, path):
    path = Path(path)
    print(f'saving buffer to {path}')
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('wb') as fi:
        pickle.dump(buffer, fi)


def load(path):
    path = Path(path)
    print(f'loading buffer from {path}')
    with path.open('rb') as fi:
        return pickle.load(fi)


class Buffer():
    def __init__(self, elements, size=64):
        self.data = {
            el: np.zeros((size, *shape), dtype=dtype)
            for el, shape, dtype in elements
        }
        self.size = size
        self.cursor = 0
        self.full = False

    def __len__(self):
        return len(self.data['observation'])

    @property
    def cursor(self):
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        if value == self.size:
            self._cursor = 0
            self.full = True
        else:
            self._cursor = value

    def append(self, data):
        for name, el in zip(data._fields, data):
            sh = self.data[name][0].shape
            self.data[name][self.cursor, :] = np.array(el).reshape(sh)
        self.cursor = self.cursor + 1

    def sample(self, num):
        if not self.full:
            raise ValueError("buffer is not full!")
        idx = np.random.randint(0, self.size, num)
        batch = {}
        for name, data in self.data.items():
            batch[name] = data[idx, :]
        return batch

    def level(self):
        return (len(self) / self.size) * 100
