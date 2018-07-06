# Spaces 

Space objects are used by both environments and agents.  This interaction with both parts of the library makes a good API very challenging and very useful.

The GlobalSpace object can be used for state, observation and action spaces.  The library design is inspired by [Open AI gym](https://github.com/openai/gym/tree/master/gym/spaces).

## Working with action spaces

Action spaces are used by agents to understand what they can do in an environment.

DQN requires a discrete action space.  The GlobalSpace object has functionality to create and sample from a discrete sample of the action space.

```python
from energy_py.common.spaces import ContinuousSpace
from energy_py.common.spaces import DiscreteSpace
from energy_py.common.spaces import GlobalSpace

#  create an action space with one discrete action and one continuous action
action_space = GlobalSpace('action').from_spaces(
    [ContinuousSpace(0, 100), DiscreteSpace(3)],
    ['acceleration', 'gear']
)

#  randomly sampling an action
action = action_space.sample()

#  check that an action is valid
assert action_space.contains(action)

#  the space accomodates sampling from a discrete version of the space
#  continuous spaces are discretized into a given number of choices
#  discrete spaces are left untouched

#  first the space is discretized into 20 choices per space dimension 
discrete_spaces = action_space.discretize(20)

#  then we can sample from a discrete representation of the space
action = action_space.sample_discrete()
```
## Working with state and observation spaces

State spaces are used by environments to understand what the current state variables are.  Observation spaces are used by agents to access infomation about the environment.
 
The GlobalSpace object has functionality for loading and sampling episodes from datasets.

```python
#  load a state or observation space from a dataset
state_space = GlobalSpace('state').from_dataset('example')

# we can sample an episode from the state
episode = state_space.sample_episode(0, 100)

# single state or observations from the current episode can be accessed by calling the space
state = state_space(steps=0)
```
