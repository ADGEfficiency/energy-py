## energy_py environments readme

All energy_py enviromnents take the following as inputs
```
lag
episode_length
episode_start
```

The lag determines the difference between the state and observation.  

State = the true value of variables such as price & demand at at time t + lag

Observation = the observed variables visible to the agent at time t
```
lag <0 can only see past
lag =0 can see present
lag >0 can see future
```
