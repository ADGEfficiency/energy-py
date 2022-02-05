import json
from pathlib import Path
import joblib
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
    path.mkdir(exist_ok=True, parents=True)
    meta = {
        'elements': buffer.elements,
        'size': buffer.size,
        'cursor_min': buffer.cursor_min,
    }
    (path / 'meta.json').write_text(json.dumps(meta))

    for name, data in buffer.data.items():
        np.save(path / f"{name}.npy", data)


def load(path):
    path = Path(path)
    print(f'loading buffer from {path}')
    meta = json.loads((path / 'meta.json').read_text())
    buf = Buffer(**meta)
    for name in buf.data.keys():
        buf.data[name] = np.load(path / f"{name}.npy")
    return buf


class Buffer():
    """
    Buffer has no concept of n_batteries
    - each sample of experience is all stored on a 'one battery' level
    """
    def __init__(
        self,
        elements,
        size=64,
        cursor_min=0
    ):
        self.elements = elements
        self.size = int(size)
        self.data = {
            el: np.zeros((self.size, *shape), dtype=dtype)
            for el, shape, dtype in elements
        }
        self.cursor = cursor_min
        self.cursor_min = cursor_min
        self.full = False

    def __len__(self):
        return len(self.data['observation'])

    @property
    def cursor(self):
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        if value == self.size:
            self._cursor = self.cursor_min
            self.full = True
        else:
            self._cursor = value

    def append(self, data):
        for name, data in data.items():
            sh = self.data[name][0].shape
            self.data[name][self.cursor, :] = np.array(data).reshape(sh)

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
