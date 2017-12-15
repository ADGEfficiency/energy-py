import numpy as np

from energy_py import DiscreteSpace, ContinuousSpace, GlobalSpace
from energy_py.agents import Memory

observation_space = GlobalSpace([ContinuousSpace(0,10)])
action_space = GlobalSpace([ContinuousSpace(50,100)])
discount = 0.9
mem = Memory(observation_space,
             action_space,
             discount,
             memory_length=10)

def method_one(rewards):
    rtns = []
    for t, transition in enumerate(rewards):
        if t == None:
            total_return = 0
        else:    
            total_return = sum(discount**i * r for i, r in enumerate(rewards[t:]))
        rtns.append(total_return)
    rtns = np.array(rtns).reshape(-1)
    return rtns

def method_two(rewards):
    rewards = list(rewards)
    rewards.reverse()
    rtns_ = np.zeros(len(rewards))
    for i, r in enumerate(rewards):
        if i == 0:
            total_return = r
        else:
            total_return = r + discount * rtns_[i-1]
        rtns_[i] = total_return

    rtns_ = list(rtns_)
    rtns_.reverse()
    rtns = np.array(rtns_).reshape(-1)
    return rtns

def test_returns_calc():
    rewards = np.ones(10)

    result_one = method_one(rewards)
    result_two = method_two(rewards)
    result_memory = mem.calculate_returns(rewards).reshape(-1)

    for v1, v2, v3 in zip(result_one, result_two, result_memory):
        assert (v1 == v2).all()
        assert (v1 == v3).all()

def test_add_exp():
    for i in range(50):
        mem.add_experience(observation_space.sample(),
                           action_space.sample(),
                           1,
                           observation_space.sample(),
                           True,
                           0,
                           0)

    assert mem.num_exp == 50 
