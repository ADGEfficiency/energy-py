## energy_py environments readme

Currently the most up-to-date environment is the battery environment.  

I am planning to build more environments
  pre/post cooling flexibility
  ice storage
  combined heat & power

All environments follow the general reinforcement learning environment schema of step() & reset()

All environments have
  action_space = list of spaces object (one per action dimension)
  observation_space = list of spaces objects (one per observation dimension)
  reward_range = a single space object with upper & lower bounds for the reward

The base class for environmnets is Base_Env

The child class for time series environments is Time_Series_Env
