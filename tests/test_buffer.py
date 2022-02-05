from collections import defaultdict
import shutil

import numpy as np

from energypy.agent.memory import Buffer, save, load

def sample():
    o = np.random.random((4, 4, 2))
    a = np.random.random((4, 3, 3))
    return o, a


def test_save_to_numpy_and_meta():
    elements = (
        ('o', (4, 4, 2), 'float32'),
        ('a', (4, 3, 3), 'float32'),
    )
    buf = Buffer(elements, size=4, cursor_min=2)

    ds = defaultdict(list)
    for _ in range(4):
        o, a = sample()
        buf.append({'o': o, 'a': a})
        ds['o'].append(o)
        ds['a'].append(a)

    assert buf.full
    save(buf, './tmp/buffer-test')

    #  nbuf = new buffer
    nbuf = load('./tmp/buffer-test')
    assert len(nbuf.elements) == 2
    assert nbuf.cursor_min == 2
    assert nbuf.full
    shutil.rmtree('./tmp/buffer-test')
