## Combined Heat and Power 

This is a non-urgent and long term project.  

Two versions of the model

1. mixed integer linear programming (MILP)
2. markov decision process

### Carbon value of CHP

The carbon value of CHP is a delta versus a no CHP case.  The no CHP case is supplying electricity from the grid and heat from steam boilers.

The carbon value of CHP is a function of

- the carbon intensity of the fuel (most commonly gas)
- the carbon intensity of grid electricity
- the efficiency of heat generation in a boiler
- the amount of heat recovered
- electrical parasitics of the CHP plant

This model is based on natural gas as the fuel in gas turbines, gas engines, steam boilers.

The carbon value of CHP in the real world depends 

### Technical

The aim of this model is to introduce all the common components of a CHP system 

- steam headers
- steam turbines
- gas turbines
- gas engines
- steam boilers
- deaeration

The aim is to build the simplest possible model with all of these components.  

In real CHP plants boiler plant will be installed in an `n+1` configuration - for example two boilers sized to meet the peak steam demand of the site.  In this model only one boiler is included for simplicity.  

In real CHP plants boiler plant will often run hot - that is sitting at minimum turn down ready to pick up load if another steam generator trips.  This minimum turndown takes away steam that could be generated elsewhere (say from the exhast gas of a gas turbine).  This minimum turndown is modelled (it is simple to change the minimum continous value for the asset by changing the constraint in the MILP model).  

### MILP

To gurantee convergence to the global optimum the equations used to describe the CHP plant must be linear.  

However, an energy balance is bi-linear!  If we try to model both a mass and energy balance (i.e. including two constraints in the MILP model) then we will end up mulitplying variables together

```
energy in = energy out

mass flow in * enthalpy in = mass flow out * enthalpy out
```

For example - consider the mass and energy balance around a steam header.  The mass balance will change as the loads on gas turbines etc change, or as demand for steam changes.  The energy balance will change as these different temperature and pressure (ie enthalpy) inputs and outputs changes. 

This means that the engineer must choose whether to balance either mass or energy.  Steam based systems (i.e. combined cycle gas turbines) can be balanced using mass.  Hot water systems (i.e. district heating) can be balanced using heat.  

Note that electrical power balances can be done alongside either mass or energy - there is no bi-linearity.

## Plants to model

Biomass boiler + steam turbine (Kinleith)
