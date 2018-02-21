"""
Tests for prioritized experience replay

ref Schaul et. al (2015) arxiv.org/pdf/1511.05952.pdf

New transitions (ie those without a TD error) are given highest priority

Define a probability of a single sample being transitioned (P)
as a function of the priority (p):

P = p^alpha / sum(p^alpha)

The priority p is defined by eithe
p = TDerror + a small constant
p = 1 / rank(TDerror)

Schaul reccomends second method.

Binary tree notes

"""

from collections import namedtuple
import numpy as np

Experience = namedtuple('Experience', ['observation',
                                       'action',
                                       'reward',
                                       'next_observation',
                                       'terminal',
                                       'priority'])

def generate_experience(p=None):
    obs = np.random.rand(1, 4)
    act = np.random.rand(1, 2)
    rew = np.random.randint(10)
    next_obs = np.random.rand(1, 2)
    terminal = False

    if p is None:
        prio = np.random.randint(10)
    else:
        prio = p

    return Experience(obs, act, rew, next_obs, terminal, prio)


def test_getChild():
    heap = MinBinaryHeap()
    priorities = [random.randint(0, 100) for _ in range(10)]
    [heap.insert(p) for p in priorities]
    i = 1
    while i < heap.currentSize + 1:
        output = heap.getChild(i)

        if isinstance(output, tuple):
            assert output[0] <= heap[i]

        i += 1

class MinBinaryHeap(object):
    """
    http://interactivepython.org/runestone/static/pythonds/Trees/BinaryHeapImplementation.html
    """

    def __init__(self):
        #  intialize heapList with a zero to make the int division simpler
        self.heapList = [0]
        self.currentSize = 0

    def insert(self, experience):
        self.heapList.append(experience)

        self.currentSize += 1

        self.percUp(self.currentSize)

    def __getitem__(self, key):
        return self.heapList[key]

    def __setitem__(self, key, item):
        self.heapList[key] = item

    def print_tree(self):
        print('printing tree')
        i = 1
        while i < self.currentSize + 1:
            output = self.getChild(i)
            if isinstance(output, str):
                print(output)
            else:
                print(output[0])
            i += 1

    def percUp(self, i):
        """
        Once a new item has been appended to the tree, this function
        positions the item properly

        args
            i (int) start position
        """

        #  while loop iterates down using floor division by 2
        #  i.e.
        #  i=10, i=5, i=2, end
        while i // 2 > 0:

            #  check if node is less than parent
            if self.heapList[i] > self.heapList[i // 2]:
                #  get the parent
                tmp = self.heapList[i // 2]
                #  replace the parent with the node
                self.heapList[i // 2] = self.heapList[i]
                #  replace the node with the parent
                self.heapList[i] = tmp

            #  move onto the next floor div of 2
            i = i // 2

    def removeMax(self):
        """
        Removes the highest priority item

        percDown() repositions the tree
        """
        #  grab out the highest priority element
        max_val = self[1]
        #  swap the root with the smallest child that is smaller than the root
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize += -1

        #  removes the element that was at self.currentSize
        self.heapList.pop()

        self.percDown(1)
        print('removed {}'.format(max_val))
        return max_val

    def percDown(self, i):
        """
        Moves an item from a node down the tree until the order is correct
        """
        #  the while loop will break when we are at a node with no children
        while (i * 2) <= self.currentSize:
            max_child, idx = self.getChild(i)

            if self[i] < max_child:
                temp = self[i]
                self[i] = max_child
                self[idx] = temp
            i = idx

    def getChild(self, i):
        """
        Finds the maximum child of a node

        args
            i (int) start position
        """
        print('getting child for node {} val {}'.format(i, self[i]))
        #  if no left
        if i * 2 > self.currentSize:
            return 'no child'

        #  if no right child exists
        #  this could be an exception ina  call to get children
        if i * 2 + 1 > self.currentSize:
            left = self[i*2]
            print('left only {}'.format(left))
            return left, i*2

        left = self[i*2]
        right = self[i*2+1]
        print('left {} right {}'.format(left, right))

        if left >= right:
            return left, i*2
        else:
            return right, i*2 + 1


if __name__ == '__main__':
    import random

    test_getChild()

    def test_removeMax():

        heap = MinBinaryHeap()
        priorities = [random.randint(0, 100) for _ in range(10)]
        [heap.insert(p) for p in priorities]
        s_priorities = sorted(priorities, reverse=True)

        print('sorted priorities {}'.format(s_priorities))
        for priority in s_priorities:
            max_p = heap.removeMax()
            assert priority == max_p

    test_removeMax()

    heap = MinBinaryHeap()
    priorities = [random.randint(0, 100) for _ in range(10)]
    [heap.insert(p) for p in priorities]

    heap.print_tree()
