##  flexibility environment

Action space is a binary action space.  
```
action = np.array([flex_on])
         shape=(1, 2)
```
Reward is the net effect of the flexibility action 
```
reward = flex_action * electricity price
$ / 5min = MW * $/MW / 12
```

Environment simulates a electricity flexiblity action.  The flexibility action is composed of three parts.

1 - flex_initial
2 - flex_final
3 - relaxation

The user can define the size [MW] of the flexibility during the initial and final stages.  The environment forces these
to be equal ad opposite.

After the intial & final stages are over a relaxation period starts.  The flex_action is zero for this period (ie reward
is zero).
