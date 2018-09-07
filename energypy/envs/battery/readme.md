## Battery environment
An environment simulating electric battery storage.

The action for this env is
```
action = np.array([charge, discharge])
         shape = (1, 2)
```

The net effect of these two actions on the battery is calculated by
```
net_charge = charge - discharge
```

The reward is the net effect of the battery on the site import/export
```
reward = -(gross_rate / 12) * electricity_price
```

The round trip efficiency of the battery is modelled by reducing the charge.

## Basic usage
A demo Jupyter Notebook is [available here.](https://github.com/ADGEfficiency/energy_py/blob/master/energy_py/main/notebooks/battery/env_demo.ipynb).

## Key technical assumptions

power_rating = maximum rate of charge & discharge

capacity = maximum amount of electricity that can be stored

round_trip_eff = efficiency of storage.  applied onto all electricity stored.
    rate = gross_rate*(1-effy)
