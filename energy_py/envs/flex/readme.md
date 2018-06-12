#  Flexibility environment
Environment to simulate a flexbile electricity asset - ie a chiller.  Also known as demand side response.

Currently two environments are implemented, which differ based on the action space.

##  flex-v0
0 = no_op
1 = start flex down -> flex up cycle
2 = start flex up -> flex down cycle

##  flex-v1
0 = no op
1 = start (if available), continue if in flex_down
2 = stop (if in flex_down cycle)

Reward is the net effect of the flexibility action, and is the same for both environments
```
reward = flex_action * electricity price
$ / 5min = MW * $/MW / 12
```

A flexibility action consists of a period of flexing up (increasing consumption), a period of decreasing consumption and
a relaxation period.

The agent can choose to start either a up/down/relax or a down/up/relax.

Once a flex action is started it runs its full cycle.

Option to apply an efficiency (the flex up becomes more than the flex down).
