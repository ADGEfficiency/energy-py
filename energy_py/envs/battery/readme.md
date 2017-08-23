## readme for all energy_py ENVS

All energy_py enviromnents take the following as inputs:
lag 
episode_length
episode_start

The lag determines the difference between the state and observation.  
State = the true value of variables such as price & demand at at time t + lag
Observation = the observed variables visible to the agent at time t

lag <0 can only see past
lag =0 can see present
lag >0 can see future

## readme for the Battery Environment

The action for this env is 
action = np.array(charge, discharge)

The net effect of these two actions on the battery is calculated by
net_charge = charge - discharge

The reward is the cost to supply electricity to site
Note the use of the negative
reward = -(customer_demand + battery_rate) * electricity_price 

Basic usage 
A demo Jupyter Notebook is available here 

Experiments are available here



Key technical assumptions

power_rating = maximum rate of charge & discharge
capacity = maximum amount of electricity that can be stored
round_trip_eff = efficiency of storage.  applied onto all electricity stored.
    rate = gross_rate*(1-effy)


