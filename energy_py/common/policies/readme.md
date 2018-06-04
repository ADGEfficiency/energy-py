#  Policies
Reinforcement learning policies - functions that map state -> action

Policies are implemented as functions that create tensorflow graphs

**epsilon_greedy_policy()**
- with a probability epsilon, either selects the optimal action (the argmax across Q(s,a)) or randomly selects an action 

**softmax_policy()**
- uses a softmax distribution over Q(s,a) to select actions

See energy_py/notebooks/softmax_policy.ipynb for an indepth look at how the softmax policy works
