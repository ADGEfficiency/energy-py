# Spaces 

Objects represent state, observation and action spaces.  Space objects are used by both environments and agents.  The energy_py Space API is inspired by [Open AI gym](https://github.com/openai/gym/tree/master/gym/spaces).

## Usage

```
from energy_py.common.spaces import ContinuousSpace
from energy_py.common.spaces import DiscreteSpace
from energy_py.common.spaces import GlobalSpace
```

# working with action spaces
```
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
#  first the space is discretized
discrete_spaces = action_space.discretize(20)

#  then we can sample from a discrete representation of the space
action = action_space.sample_discrete()
```
# working with state and observation spaces
```
#  load a state or observation space from a dataset
state_space = GlobalSpace('state').from_dataset('example')

# we can sample an episode from the state
episode = state_space.sample_episode(0, 100)

# single state or observations can be accessed by calling the space
state = state_space(steps=0)
```
