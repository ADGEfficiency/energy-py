"""
Creating a sum tree for use in prioritized experience replay

Priorities are stored as leaf (ie terminal) nodes with internal nodes
containing the sums


Reference = Open AI Baselines binary tree implementations:
baselines/common/segment_tree.py
"""
import random

from collections import namedtuple

from energy_py.agents import Memory
from energy_py import SumTree, MinTree


def test_rememeber():
    mem = PrioritizedReplay(10)

    space = {'10': 5,
             '5': 2,
             '1': -1}

    for _ in range(30):
        e = random.choice(list(space.keys()))
        p = space[e]

        exp = {'experience': e, 'priority': p}
        mem.remember(**exp)

    for idx, exp in enumerate(mem.experiences):
        priority = mem.sumtree[idx]
        expected_priority = space[exp]
        assert priority == expected_priority

def test_trees():
    """
    Tests the sum and min operations over the memory
    """
    mem = PrioritizedReplay(10)

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


class PrioritizedReplay(Memory):
    def __init__(self, size, obs_shape, action_shape):

        super(PrioritizedReplay, self).__init__(observation_space_shape,
                                          action_space_shape,
                                          size)
        self.size = size
        self.experiences = []
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

    def __repr__(self):
        return '<class Memory len={}>'.format(len(self))

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        return self.experiences[idx]

    def remember(self, experience, priority=None):
        """
        Experience is added to the experiences list at self._next_index

        self.experiences is a fixed length length - older experiences are
        wiped over using _next_index

        New experience should be saved at max priority.  priority arg is
        used for testing

        """
        #  save the experience into experiences
        #  if we are still filling up the experiences, append onto the list
        if self._next_index >= len(self.experiences):
            self.experiences.append(experience)

        #  else we replace an old experience
        else:
            self.experiences[self._next_index] = experience

        #  save the priority in the sumtree
        #  new experiences default at max priority so they always get
        #  trained on at least once
        if priority is None:
            priority = self.max_priority ** self.alpha

        self.sumtree[self._next_index] = priority
        self.mintree[self._next_index] = priority

        #  modulus gives us the index of the next oldest experience
        self._next_index = (self._next_index + 1) % self.size

        print('{} len experiences after adding exp next oldest is {}'.format(len(self.experiences),
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
            mass = random.random() * self.sumtree.sum(0, len(self.experiences)-1)
            idx = self.sumtree.find(mass)
            batch_idx.append(idx)

        return batch_idx


if __name__ == '__main__':
    test_rememeber()
    test_trees()

    mem = PrioritizedReplay(10)

    exp, pr = [], []
    for _ in range(30):
        e = random.randint(0, 100)
        p = random.random()

        exp.append(e)
        pr.append(p)

        mem.remember(e, p)

    #  Schaul suggests a, b = 0.7, 0.5 for rank
    #  a,b = 0.6, 0.4 for proportional

    #  beta controls the bias correction done by importance sampling
    batch_size = 10
    beta = 1
    import random
    from collections import defaultdict

    sample_size = min(batch_size, len(mem))
    batch_dict = defaultdict(list)

    #  get indexes for a batch sampled using the priorities
    indexes = mem.sample_proportional(10)
    batch = [mem[idx] for idx in indexes]

    #  the probability of sampling is defined as
    #  P = priority / sum(priorities)
    #  first calculate the minimum probability for the tree priorities
    #  equn 1 Schaul (2015)
    p_min = mem.mintree.min() / mem.sumtree.sum()

    #  equn 2 Schaul (2015)
    max_weight = (1 / (p_min * len(mem.experiences))) ** beta

    weights = []
    for idx in indexes:
        sample_probability = mem.sumtree[idx] / mem.sumtree.sum()

        weight = (1 / (sample_probability * len(mem.experiences))) ** beta

        weights.append(weight/max_weight)

    for exp in batch:
        # batch_dict['observations'].append(exp.observation)
        # batch_dict['actions'].append(exp.action)
        # batch_dict['rewards'].append(exp.reward)
        # batch_dict['next_observations'].append(exp.next_observation)
        # batch_dict['terminal'].append(exp.terminal)
        batch_dict['exp'].append(exp)
    # for key, data in batch_dict.items():
    #     batch_dict[key] = np.array(data).reshape(-1, *self.shapes[key])



