import numpy as np
from energy_py import Standardizer, Normalizer
eps = 1e-5
batch = np.array([[1,2,-3],
                  [0,-2,0],
                  [4,5,-1]]).reshape(3,-1)

def test_std():
    std = Standardizer(length=3, use_history=False)
    res = std.transform(batch)

    #  check that the array is shaped correctly
    assert res.shape == batch.shape

    #  check our means are close to 0 
    assert np.any(np.absolute(res.mean(axis=0) < eps))
    #  check our standard deviations are close to 1
    assert np.any(np.absolute(res.std(axis=0) - 1 < eps))

def test_norm():    
    norm = Normalizer(length=3, use_history=False)
    res = norm.transform(batch)

    #  check that the array is shaped correctly
    assert res.shape == batch.shape

    #  check our means are close to 0 
    assert np.any(np.absolute(res.max(axis=0) - 1 < eps))
    #  check our standard deviations are close to 1
    assert np.any(np.absolute(res.min(axis=0) < eps))

def test_std_hist():
    std = Standardizer(length=3, use_history=True)
    res = std.transform(batch)
    old_mean, old_stdev = std.means, std.stdevs

    res = std.transform(batch)
    new_mean, new_stdev = std.means, std.stdevs
    assert np.all(new_mean == old_mean)
    assert np.all(new_stdev == old_stdev)

    res = std.transform(batch*2)
    new_mean, new_stdev = std.means, std.stdevs
    assert np.all(new_mean != old_mean)
    assert np.all(new_stdev != old_stdev)

def test_norm_hist():
    norm = Normalizer(length=3, use_history=True)
    res = norm.transform(batch)
    old_max, old_min = norm.maxs, norm.mins

    res = norm.transform(batch)
    new_max, new_min = norm.maxs, norm.mins
    assert np.all(new_max == old_max)
    assert np.all(new_min == old_min)

