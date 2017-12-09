In reinforcement learning we want to approximate various functions
 - value function V(s)
 - Q(s,a) with a single output node
 - Q(s,a) with one node per action
 - a Gaussian policy
 - a determinsitic policy

 The idea behind the use of the function approximator class is that it 
 will allow me to wrap any deep learning library - keeping energy_py
 deep learning library agnostic.

 Currently building function approximators in TensorFlow.  Starting with
 keeping them in separate classes - will think about using a parent class
 once I can see how much code is repeated.
