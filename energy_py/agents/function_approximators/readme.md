This folder contains function approximators.

In reinforcement learning often want to approximate three functions:
 - value function V(s)
 - action-value function Q(s,a)
 - a policy pi(s)

 The idea behind the use of the function approximator class is that it 
 will allow me to wrap any deep learning library - keeping energy_py
 deep learning library agnostic.
