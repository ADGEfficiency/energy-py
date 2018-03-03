import numpy as np

from energy_py import Normalizer, Standardizer

#  make a batch with one dimensional samples
batch = np.random.rand(10, 6)

def test_normalizer():
    norm = Normalizer(use_history=True)
    
    _ = norm.transform(batch[1:5])
    _ = norm.transform(batch[5:])

    assert norm.maxs.all() == np.max(batch, axis=0).all()

def test_standardizer():
    std = Standardizer()

    _ = std.transform(batch[1:5])
    _ = std.transform(batch[5:])

    assert std.means.all() == np.mean(batch, axis=0).all()
    assert std.stds.all() == np.std(batch, axis=0).all()
