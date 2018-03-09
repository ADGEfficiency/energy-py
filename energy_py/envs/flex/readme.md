##  Flexibility environment
Environment to simulate a flexbile electricity asset - ie a chiller.  Also known as demnad side response.

Action space is a discrete action space.  
```
action = np.array([flex_action])
    action = 0 -> no op
    action = 1 -> flex up then down
    action = 2 -> flex down then up
```
Reward is the net effect of the flexibility action 
```
reward = flex_action * electricity price
$ / 5min = MW * $/MW / 12
```

A flexibility action consists of a period of flexing up (increasing consumption), a period of decreasing consumption and
a relaxation period.

The agent can choose to start either a up/down/relax or a down/up/relax.

Once a flex action is started it runs its full cycle.

Option to apply an efficiency (the flex up becomes more than the flex down).
