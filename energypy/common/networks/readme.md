# Networks 

Code make tensorflow neural networks.  Designed to be generic (i.e. not reinforcement learning specific)


To correctly name the variables and still allow variable sharing:
```
with tf.name_scope('online_network):
    layer = fully_connected_layer('input_layer', ...)
```

**layers.fully_connected_layer()**
- Creates a single fully connected layer
- Can use either a relu or linear activation function

The layer is created using `tf.get_variable` to allow variable sharing using `scope.reuse_variables()`.  [energy_py/notebooks/tf_variable_sharing.ipynb](https://github.com/ADGEfficiency/energy_py/blob/master/notebooks/tf_variable_sharing.ipynb) for an indepth look.

**networks.feed_forward()**
- Creates a feedforward neural network (aka multi layer perceptron)
- Currently no support for batch or layer norm
