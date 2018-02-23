"""
Creating a sum tree for use in prioritized experience replay

Priorities are stored as leaf (ie terminal) nodes with internal nodes
containing the sums


Reference = Open AI Baselines binary tree implementations:
baselines/common/segment_tree.py
"""
import random

from collections import namedtuple
Args = namedtuple('args', ['start',
                       'end',
                       'node',
                       'node_start',
                       'node_end'])

def test_rememeber():
    mem = PrioritizedReplayMemory(10)

    space = {'10': 5,
             '5': 2,
             '1': -1}

    for _ in range(30):
        e = random.choice(list(space.keys()))
        p = space[e]

        exp = {'experience': e, 'priority': p}
        mem.remember(**exp)

    for idx, exp in enumerate(mem.storage):
        priority = mem.sumtree[idx]
        expected_priority = space[exp]
        assert priority == expected_priority

def test_trees():
    """
    Tests the sum and min operations over the memory
    """
    mem = PrioritizedReplayMemory(10)

    exp, pr = [], []
    for _ in range(30):
        e = random.randint(0, 100)
        p = random.random()

        exp.append(e)
        pr.append(p)

        mem.remember(e, p)

    sumtree = mem.sumtree
    mintree = mem.mintree

    s1 = sumtree.sum()
    m1 = mintree.min()

    tol = 1e-6
    assert s1 - sum(pr[-10:]) < tol
    assert m1 - min(pr[-10:]) < tol


class SegmentTree(object):

    def __init__(self, capacity, operator, neutral_element):

        self.capacity = capacity
        self.operation = operator

        self.values = [neutral_element for _ in range(2 * self.capacity)]

    def __setitem__(self, idx, val):
        """
        Adds a priority as a leaf node
        """

        #  the index for the priority
        idx += self.capacity

        #  use this index to save the priority as a leaf node
        self.values[idx] = val

        #  move up to the parent
        #  in a binary heap the parent is located at n/2
        idx //= 2

        #  while loop runs as we back up the tree
        while idx >= 1:
            #  we set the value for the parent node using the two children
            #   left=2*p, right=2p+1
            self.values[idx] = self.operation([self.values[2 * idx],
                                               self.values[2 * idx + 1]])

            #  move up to the next level of the tree (the next parent)
            idx //= 2

    def __getitem__(self, idx):
        """
        Gets an item using the index of the memory (not of this tree!)
        """
        #  check that index is less than the memory capacity
        assert 0 <= idx < self.capacity
        #  return the priority for that node
        return self.values[self.capacity + idx]

    def reduce(self, start, end):
        #  checks on end TODO
        if end is None:
            end = self.capacity

        end -= 1

        args = Args(start, end, node=1, node_start=0, node_end=self.capacity-1) 
        print('initial reduce call')
        print(args)
        return self._reduce_helper(**args._asdict())

    def _reduce_helper(self, start, end, node, node_start, node_end):

        if start == node_start and end == node_end:
            print('CONDITION ONE')
            return self.values[node]

        #  find the middle node (factor of two)
        mid = (node_start + node_end) // 2
        print('mid {}'.format(mid))
        if end <= mid:
            #  move to node to left node and node_end to mid
            args = Args(start, end, node=2*node,
                        node_start=node_start, node_end=mid)
            print('COND TWO')
            print(args)
            return self._reduce_helper(**args._asdict())

        else:
            #  middle node + 1 - is this 2p+1 ???
            if mid + 1 <= start:
                #  we are moving onto the right node here
                args = Args(start, end, node=2*node+1,
                            node_start=mid+1, node_end=node_end)
                print('COND THREE')
                print(args)
                return self._reduce_helper(**args._asdict())

            else:
                arg1 = Args(start, mid, 2*node, node_start, mid)
                arg2 = Args(mid+1, end, 2*node+1, mid+1, node_end)
                print('COND FOUR')
                print(arg1)
                print(arg2)
                return self.operation([self._reduce_helper(**arg1._asdict()),
                            self._reduce_helper(**arg2._asdict())])


class MinTree(SegmentTree):
    def __init__(self, capacity):
        super(MinTree, self).__init__(
            capacity=capacity,
            operator=min,
            neutral_element=float('inf'))

    def min(self, start=0, end=None):
        return super(MinTree, self).reduce(start, end)


class SumTree(SegmentTree):
    def __init__(self, capacity):
        super(SumTree, self).__init__(
            capacity=capacity,
            operator=sum,
            neutral_element=0.0)

    def sum(self, start=0, end=None):
        return super(SumTree, self).reduce(start, end)

    def find(self, prob):
        """
        Highest index such that sum[0:i-1] <= prob

        If values are probabilities (ie <1) then this function can be
        used to sample according to the discrete probability efficiently

        args
            prob (float)

        return
            idx (int) highest index that satasifies the probability constraint
        """
        assert 0 <= prob <= self.sum(start=0, end=self.capacity) + 1e-5

        #  start the index counter at the top of the tree
        idx = 1

        #  while non-leaf
        while idx < self.capacity:
            #  if the left node is greater than our probability
            #  move to the left node
            if self.values[2*idx] > prob:
                idx = 2 * idx
            else:
                #  otherwise, move to the right index
                prob -= self.values[2*idx]
                idx = 2 * idx + 1

        return idx - self.capacity


class PrioritizedReplayMemory(object):
    def __init__(self, size):
        self.size = size
        self.storage = []
        #  while loop to set the tree capacity with a factor of two
        tree_capacity = 1
        while tree_capacity < self.size:
            tree_capacity *= 2

        self.sumtree = SumTree(tree_capacity)
        self.mintree = MinTree(tree_capacity)
        #  TODO mintree

        self._next_index = 0
        self.max_priority = 1
        self.alpha = 2

    def __getitem__(self, idx):
        return self.storage[idx]

    def remember(self, experience, priority=None):
        """
        Experience is added to the storage list at self._next_index

        self.storage is a fixed length length - older experiences are
        wiped over using _next_index

        New experience should be saved at max priority.  priority arg is
        used for testing

        """
        #  save the experience into storage
        #  if we are still filling up the storage, append onto the list
        if self._next_index >= len(self.storage):
            self.storage.append(experience)

        #  else we replace an old experience
        else:
            self.storage[self._next_index] = experience

        #  save the priority in the sumtree
        #  new experiences default at max priority so they always get
        #  trained on at least once
        if priority is None:
            priority = self.max_priority ** self.alpha

        self.sumtree[self._next_index] = priority
        self.mintree[self._next_index] = priority

        #  modulus gives us the index of the next oldest experience
        self._next_index = (self._next_index + 1) % self.size

        print('{} len storage after adding exp next oldest is {}'.format(len(self.storage),
                                                                         self._next_index))

    def get_batch(self, batch_size, beta):
        """
        Samples a batch of experiences

        args
            batch_size (int)
            beta (float) determines strength of importance weights
                0 -> no correction, 1 -> full correction
        """
        beta = float(beta)
        assert 0 < beta <= 1.0

        #  get indexes for a batch sampled using the priorities
        indexes = self.sample_proportional(batch_size)

        #  the probability of sampling is defined as
        #  P = priority / sum(priorities)
        #  first calculate the minimum probability for the tree priorities
        #  equn 1 Schaul (2015)
        p_min = self.mintree.min() / self.sumtree.sum()

        #  equn 2 Schaul (2015)
        # max_weight = 

    def sample_proportional(self, batch_size):
        """

        """
        batch_idx = []

        for _ in range(batch_size):
            mass = random.random() * self.sumtree.sum(0, len(self.storage)-1)
            idx = self.sumtree.find(mass)
            batch_idx.append(idx)

        return batch_idx


if __name__ == '__main__':
    test_rememeber()
    test_trees()

    mem = PrioritizedReplayMemory(10)

    exp, pr = [], []
    for _ in range(30):
        e = random.randint(0, 100)
        p = random.random()

        exp.append(e)
        pr.append(p)

        mem.remember(e, p)

    sumtree = mem.sumtree
    mintree = mem.mintree

    #  the sample proportonal call of sum()
    s = sumtree.sum(start=0, end=len(mem.storage) - 1)

    #  to calculate p_min we need to
    #  call sum with defaults, ie start=0, end=None
    s1 = sumtree.sum()
    m1 = mintree.min()

    tol = 1e-6
    assert s1 - sum(pr[-10:]) < tol
    assert m1 - min(pr[-10:]) < tol

    #  Schaul suggests a, b = 0.7, 0.5 for rank
    #  a,b = 0.6, 0.4 for proportional

    #  beta controls the bias correction done by importance sampling
    #  
    beta = 1

    #  get indexes for a batch sampled using the priorities
    indexes = mem.sample_proportional(10)

    #  the probability of sampling is defined as
    #  P = priority / sum(priorities)
    #  first calculate the minimum probability for the tree priorities
    #  equn 1 Schaul (2015)
    p_min = mem.mintree.min() / mem.sumtree.sum()

    #  equn 2 Schaul (2015)
    max_weight = (1 / (p_min * len(mem.storage))) ** beta

    weights = []
    for idx in indexes:
        sample_probability = mem.sumtree[idx] / mem.sumtree.sum()

        weight = (1 / (sample_probability * len(mem.storage))) ** beta

        weights.append(weight/max_weight)




