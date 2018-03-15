"""
Prioritized experience replay
[Schaul et. al (2015) Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

Some implementations used a binary heap search tree.  The Python standard library has a [collection of functions for heap queues](https://docs.python.org/3/library/heapq.html).

[General intro to binary heaps with some Python implementation from scratch](http://interactivepython.org/runestone/static/pythonds/Trees/BinaryHeapImplementation.html).

[Takoika/PrioritizedExperienceReplay implementation](https://github.com/takoika/PrioritizedExperienceReplay/blob/master/sum_tree.py).

[TensorForce implementation](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/core/memories/prioritized_replay.py)

[Slide 20 of 'Deep Reinforcment Learning in TensorFlow'](http://web.stanford.edu/class/cs20si/lectures/slides_14.pdf) - samples using log-probabilities (not a search tree).

Open AI Baselines implementation:
[sum tree](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py),
[the memory object](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) and
[using the memory in DQN](https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py).
"""

from collections import defaultdict
import random
import logging

import numpy as np

from energy_py.agents import Experience, Memory
from energy_py import SumTree, MinTree

logger = logging.getLogger(__name__)


class PrioritizedReplay(Memory):

    """
    args
        size (int)
        obs_shape (tuple) used to reshape the observation np.array
        act_shape (tuple) used to reshape the action np.array
        alpha (float) controls prioritization
            0->no prioritization, 1-> full priorization
            default of 0.6 or 0.7 suggested in Schaul et. al (2016)
            Hessel et. al (2017) Rainbow reccomends 0.5

    """
    def __init__(self, size, obs_shape, action_shape, alpha=0.5):

        super(PrioritizedReplay, self).__init__(size,
                                                obs_shape,
                                                action_shape)
        #  use a list to store experiences
        self.experiences = []
        #  while loop to set the tree capacity as a factor of two
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


    def get_batch(self, batch_size, beta):
        """
        Samples a batch of experiences

        Schaul suggests linearly decaying beta

        args
            batch_size (int)
            beta (float) determines strength of importance weights
                beta is usually scheduled from 0.4 to 1.0 over the expt length
                0 -> no correction, 1 -> full correction
        """
        #  beta controls the bias correction done by importance sampling
        beta = float(beta)
        assert 0 < beta <= 1.0
        logger.debug('getting batch with beta {}'.format(beta))
        sample_size = min(batch_size, len(self))

        ###  get indexes for a batch sampled using the priorities
        #  these indexes are for the memory (not the tree!)
        indexes = self.sample_proportional(sample_size)
        logger.debug('indexes {}'.format(indexes))
        batch = [self[idx] for idx in indexes]

        #  the probability of sampling is defined as
        #  P = priority / sum(priorities)
        #  first calculate the minimum probability for the tree priorities
        #  equn 1 Schaul (2015)
        p_min = self.mintree.min() / self.sumtree.sum()

        #  equn 2 Schaul (2015)
        max_weight = (1 / (p_min * len(self.experiences))) ** beta

        weights = []
        logging.debug('p min {} max weight {}'.format(p_min, max_weight))
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

        #  add on the indicies for these samples and the importance weights
        batch_dict['indexes'] = indexes
        batch_dict['importance_weights'] = weights

        for key, data in batch_dict.items():
            batch_dict[key] = np.array(data).reshape(-1, *self.shapes[key])

        return batch_dict

    def sample_proportional(self, batch_size):
        """
        Because our sumtree is summing priorities, we can sample from it
        using a cumulative probability

        args
            batch_size (int)
        """
        batch_idx = []
        logger.debug('sample proportional call')
        total_mass = self.sumtree.sum(0, len(self)-1)
        for _ in range(batch_size):
            #  find a probability to sample with
            mass = random.random() * total_mass
            #  find the index for this probability
            idx = self.sumtree.find(mass)
            batch_idx.append(idx)

        return batch_idx

    def update_priorities(self, indicies, td_errors):
        """
        After learning the TD error (and therefore the priority) will change

        args
            indicies (list)
            td_errors (list)
        """
        #  cleaning up the td errors
        priorities = np.abs(td_errors) + 1e-6

        indicies = indicies.flatten().tolist()
        priorities = priorities.flatten().tolist()

        logger.debug('updating priorities i {} p {}'.format(indicies, priorities))
        assert len(indicies) == len(priorities)

        for idx, priority in zip(indicies, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            #  update the priorities in our trees
            self.sumtree[idx] = priority ** self.alpha
            self.mintree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

        logger.debug('finished updating prorities - new max priority {}'.format(self.max_priority))
