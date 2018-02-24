"""
energy_py implementation of a SumTree and MinTree.
Used to efficiently sample from a prioritized experience replay memory.

refs
    Schaul et. al (2016) Prioritized Experience Replay

inspiration for this code comes from a few different libraries
    Open AI baselines - segment_tree.py and replay_buffer.py
    http://interactivepython.org/runestone/static/pythonds/Trees/BinaryHeapImplementation.html
    TensorForce impelmentation in prioritized_replay.py

This code is tested at the same time as the PrioritizedReplay memory.
See tests/test_priority.py






"""
Args = namedtuple('args', ['start',
                       'end',
                       'node',
                       'node_start',
                       'node_end'])


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
