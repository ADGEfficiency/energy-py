import numpy as np

from energy_py import Normalizer, Standardizer
from energy_py import ContinuousSpace, DiscreteSpace, GlobalSpace

#  create a Global Space for the Normalizer test
spc = GlobalSpace([ContinuousSpace(0,10), DiscreteSpace(1), ContinuousSpace(0,10)])

#  create arrays to process
b1 = np.random.rand(10,3)
b2 = np.random.rand(10,3)
all_b = np.concatenate([b1, b2], axis=0)

def test_standardizer():
    #  test using batch statistics
    processor = Standardizer(b1.shape[1], use_history=False)
    _ = processor.transform(b1)
    _ = processor.transform(b2)
    
    assert processor.means.flatten().all() == np.mean(b2, axis=0).all()
    
def test_standardizer_hist():
    #  test using history
    true_means = np.mean(b1, axis=0)
    true_stds = np.std(b1, axis=0)
    
    processor = Standardizer(b1.shape[1], use_history=True)
    _ = processor.transform(b1)
    
    assert true_means.all() == processor.means.all()
    assert true_stds.all() == processor.stds.all()

def test_normalizer():
    #  testing using batch statistics
    processor = Normalizer(b1.shape[1], use_history=False)
    
    _ = processor.transform(b1)
    assert processor.mins.all() == np.min(b1, axis=0).all()
    assert processor.maxs.all() == np.max(b1, axis=0).all()
    
    _ = processor.transform(b2)
    assert processor.mins.all() == np.min(b2, axis=0).all()
    assert processor.maxs.all() == np.max(b2, axis=0).all()   

def test_normalizer_hist():
    #  test using history
    processor = Normalizer(b1.shape[1], use_history=True)
    
    _ = processor.transform(b1)   
    _ = processor.transform(b2)

    assert processor.mins.all() == np.min(all_b, axis=0).all()
    assert processor.maxs.all() == np.max(all_b, axis=0).all()   
    
def test_normalizer_space():
    #  test using a GlobalSpace
    processor = Normalizer(b1.shape[1], space=spc)
    
    _ = processor.transform(b1)
    _ = processor.transform(b2)
    
    assert processor.mins.all() == spc.low.all()
    assert processor.maxs.all() == spc.high.all()
