"""
Creating a sum tree for use in prioritized experience replay

Priorities are stored as leaf (ie terminal) nodes with internal nodes
containing the sums


Reference = Open AI Baselines binary tree implementations:
baselines/common/segment_tree.py
"""
import random

from collections import defaultdict, namedtuple

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
    """
    args
        size (int)


        alpha (float) controls prioritization
            0->no prioritization, 1-> full priorization
            default of 0.7 as suggested in Schaul et. al (2016)

    """
    def __init__(self, size, obs_shape, action_shape, alpha=0.7):

        super(PrioritizedReplay, self).__init__(size,
                                                obs_shape,
                                                action_shape)
        self.experiences = []

        #  while loop to set the tree capacity with a factor of two
        #  this could be put into the tree init
        tree_capacity = 1
        while tree_capacity < self.size:
            tree_capacity *= 2

        #  create the trees
        self.sumtree = SumTree(tree_capacity)
        self.mintree = MinTree(tree_capacity)

        #  _next_index controls where we put new experiences
        self._next_index = 0

        #  set the initial max priority
        self.max_priority = 1

        #  alpha controls prioritization
        self.alpha = float(alpha)
        assert self.alpha > 0

    def __repr__(self):
        return '<class PrioritizedReplayMemory len={}>'.format(len(self))

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

        #  send the memory index into the trees
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
        #  beta controls the bias correction done by importance sampling
        beta = float(beta)
        assert 0 < beta <= 1.0

        #  Schaul suggests a, b = 0.7, 0.5 for rank
        #  a,b = 0.6, 0.4 for proportional

        sample_size = min(batch_size, len(mem))
        batch_dict = defaultdict(list)

        #  get indexes for a batch sampled using the priorities
        #  these indexes are for the memory (not the tree!)
        indexes = mem.sample_proportional(sample_size)
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

        return batch_dict

    def sample_proportional(self, batch_size):
        """
        Because our sumtree is summing priorities, we can sample from it
        using a cumulative probability
        """
        batch_idx = []

        total_mass = self.sumtree.sum(0, len(self)-1)
        for _ in range(batch_size):
            #  find a probability to sample with
            mass = random.random() * total_mass
            #  find the index for this probability
            idx = self.sumtree.find(mass)
            batch_idx.append(idx)

        return batch_idx

    def update_priorities(self, indicies, priorities):
        """
        After learning the TD error (and therefore the priority) will change

        args
            indicies (list)
            priorities (list)
        """
        assert len(indicies) == len(priorities)

        for idx, priority in zip(indicies, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sumtree[idx] = priority ** self.alpha
            self.mintree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)


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


