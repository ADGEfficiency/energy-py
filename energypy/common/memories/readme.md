#  Memories

Memory structures to hold an agent's experiences

The memory remember() method is called from the agent 

Allows the agent to preprocess dimensions of experience (i.e. reward clip) before the experience is remembered

```
class Agent

    def remember(self, observation, action, reward, next_observation, done):

        if self.min_reward and self.max_reward:
            reward = max(self.min_reward, min(reward, self.max_reward))

        return self.memory.remember(observation, action, reward,
                                    next_observation, done)
```


**calculate_returns()**
- function to calculate the Monte Carlo discounted return

**class Memory**
- the base class for memories
- Experience namedtuple is used to hold a single sample of experience

**class DequeMemory**
- is the fastest impelmentation
- uses a deque to store experience as namedtuples (one per step)
- sampling by indexing experience and unpacking into arrays

**class ArrayMemory**
- stores each dimension of experience (state, action etc)
  in separate numpy arrays
- sampling experience is done by indexing each array

**class PrioritizedReplay**
- implementation of prioritized experience replay

[Schaul et. al (2015) Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

[TensorForce implementation](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/core/memories/prioritized_replay.py)

[Slide 20 of 'Deep Reinforcment Learning in TensorFlow'](http://web.stanford.edu/class/cs20si/lectures/slides_14.pdf) - samples using log-probabilities (not a search tree)

Open AI Baselines implementation
[sum tree](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py),
[the memory object](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) and
[using the memory in DQN](https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py).
