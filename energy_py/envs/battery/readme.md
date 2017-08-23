## Readme for the Battery Environment

The action for this env is
```
action = np.array(charge, discharge)
```

The net effect of these two actions on the battery is calculated by
```
net_charge = charge - discharge
```

The reward is the cost to supply electricity to site
```
reward = -(customer_demand + battery_rate) * electricity_price
```
## Basic usage
A demo Jupyter Notebook is [available here.](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/notebooks/battery/env_demo.ipynb).

## Key technical assumptions

power_rating = maximum rate of charge & discharge

capacity = maximum amount of electricity that can be stored

round_trip_eff = efficiency of storage.  applied onto all electricity stored.
    rate = gross_rate*(1-effy)
