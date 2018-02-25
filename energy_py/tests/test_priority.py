"""
Creating a sum tree for use in prioritized experience replay


References
Open AI Baselines
    baselines/common/segment_tree.py
    baselines/deepq/replay_buffer.py
"""
import random

from collections import defaultdict, namedtuple
import numpy as np

from energy_py import Experience, SumTree, MinTree, calculate_returns
from energy_py.agents import Memory


def generate_experience(obs_shape, action_shape):
    """
    A function to generate namedtuples of experience
    """
    obs = np.random.rand(1, *obs_shape)
    act = np.random.rand(1, *action_shape)
    rew = np.random.randint(10)
    next_obs = np.random.rand(1, *obs_shape)
    terminal = False

    return Experience(obs, act, rew, next_obs, terminal)


def setup_memory(size, num_exps, alpha=0.7):
    obs_shape, action_shape = (4,), (2,)
    mem = PrioritizedReplay(10, obs_shape, action_shape, alpha)
    exps = [generate_experience(obs_shape, action_shape) for _ in range(5)]
    return mem, exps


def test_calc_returns():
    rews = [10, 15, -4, 8, -1]
    discount = 1
    rtn = -1 + discount*8 + discount**2*-4 + discount**3*15 + discount**4*10

    test_rtn = calculate_returns(rews, discount)

    assert test_rtn[0] == rtn


def test_remember():
    """
    Checks the priorities are stored correctly
    """
    mem, exps = setup_memory(10, 5)
    pr = [random.random() for _ in range(len(exps))]

    #  code relies on the memory size being longer than the exps
    assert len(exps) < mem.size

    for e, p in zip(exps, pr):
        print(e)
        mem.remember(*e._asdict(), priority=p)

    for idx, e in enumerate(mem.experiences):
        p = mem.sumtree[idx]
        expected = pr[idx]

        assert p == expected


def test_trees():
    """
    Tests the sum and min operations over the memory
    """
    mem, exps = setup_memory(10, 5)

    pr = [random.random() for _ in range(len(exps))]

    for e, p in zip(exps, pr):
        mem.remember(*e._asdict(), priority=p)

    sumtree = mem.sumtree
    mintree = mem.mintree

    s1 = sumtree.sum()
    m1 = mintree.min()

    tol = 1e-6
    assert s1 - sum(pr[-10:]) < tol
    assert m1 - min(pr[-10:]) < tol


def test_update_priorities():
    mem, exps = setup_memory(10, 5, alpha=1.0)

    for exp in exps:
        #  remember experience using the default
        mem.remember(*exp._asdict().values())

    assert mem.sumtree.sum() == 5

    #  get a batch
    batch, indicies = mem.get_batch(2, beta=1)

    td_errors = np.array([0.1, 100]).reshape(2, 1)

    mem.update_priorities(indicies, td_errors)
    assert mem.mintree.min() == 0.1
    assert mem.sumtree.sum() == 100 + 3 + 0.1


class PrioritizedReplay(Memory):
    """
    args
        size (int)
        obs_shape (tuple) used to reshape the observation np.array
        act_shape (tuple) used to reshape the action np.array
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

    def remember(self, observation, action, reward,
                 next_observation, terminal, priority=None):
        """
        Adds experience to the memory

        Experience is added to the experiences list at self._next_index

        self.experiences is a fixed length length - older experiences are
        wiped over using _next_index

        New experience should be saved at max priority.  priority arg is
        used for testing

        args
            observation
            action
            reward
            next_observation
            terminal
            priority
        """
        #  create an experience named tuple
        #  add the experience to our deque
        #  the deque automatically keeps our memory at the correct size
        experience = Experience(observation,
                                action,
                                reward,
                                next_observation,
                                terminal)

        #  save the experience into experiences
        #  if we are still filling up the experiences, append onto the list
        if self._next_index >= len(self):
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

        Schaul suggests linearly decaying beta

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

        sample_size = min(batch_size, len(self))

        #  get indexes for a batch sampled using the priorities
        #  these indexes are for the memory (not the tree!)
        indexes = self.sample_proportional(sample_size)
        batch = [self[idx] for idx in indexes]

        #  the probability of sampling is defined as
        #  P = priority / sum(priorities)
        #  first calculate the minimum probability for the tree priorities
        #  equn 1 Schaul (2015)
        p_min = self.mintree.min() / self.sumtree.sum()

        #  equn 2 Schaul (2015)
        max_weight = (1 / (p_min * len(self.experiences))) ** beta

        weights = []
        for idx in indexes:
            sample_probability = self.sumtree[idx] / self.sumtree.sum()

            weight = (1 / (sample_probability * len(self.experiences))) ** beta

            weights.append(weight/max_weight)

        batch_dict = defaultdict(list)
        for exp in batch:
            batch_dict['observations'].append(exp.observation)
            batch_dict['actions'].append(exp.action)
            batch_dict['rewards'].append(exp.reward)
            batch_dict['next_observations'].append(exp.next_observation)
            batch_dict['terminal'].append(exp.terminal)

        for key, data in batch_dict.items():
            batch_dict[key] = np.array(data).reshape(-1, *self.shapes[key])

        return batch_dict, indexes

    def sample_proportional(self, batch_size):
        """
        Because our sumtree is summing priorities, we can sample from it
        using a cumulative probability

        args
            batch_size (int)
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
        print('updating priorities i {} p {}'.format(indicies, priorities))
        assert len(indicies) == len(priorities)

        for idx, priority in zip(indicies, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sumtree[idx] = priority ** self.alpha
            self.mintree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

        print('finished updating prorities - new max priority {}'.format(self.max_priority))

if __name__ == '__main__':
    test_remember()
    test_trees()

