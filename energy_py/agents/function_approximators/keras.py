import numpy as np

from keras import backend as K
from keras.layers import Dense, Activation
import keras.models
from keras.optimizers import Adam

def huber_loss(y_true, y_pred):
    """
    Implementation of the Huber loss function.
    Vlad Mnih reccomends using this (ref = 2017 Deep RL Bootcamp)

    Tour of Gotcchas when implementing DQN with Keras

    https://github.com/fchollet/keras/blob/master/keras/losses.py

    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/losses/losses_impl.py
    """
    clip_delta=1

    error = K.abs(y_true - y_pred)
    quad = K.minimum(error, clip_delta)
    return 0.5 * K.square(quad) + clip_delta * (error - quad)


class KerasFunctionApproximator(object):
    """
    The base class for Keras energy_py function approximators

    args
        model_dict (dict) : used to setup the Keras model
                            args depend on the model being built
                            {
                             'type'       : (str)
                             'input_dim   : (int) dimensions of the input lyr
                             'layers'     : (list) nodes per hidden layer
                             'output_dim' : (int) dimensions of the output lyr,
                             'lr'         : (float) learning rate
                             'batch_size  : (int) size of experience replay batch
                             'epochs'     : (int) number of passes over each batch
                            }
    """
    def __init__(self, model_dict):
        self.model_dict = model_dict

        if model_dict['type'] == 'feedforward':
            self.clipnorm = 5
            self.model = self.build_feedforward(model_dict)

    def build_feedforward(self, model_dict):
        """
        Creates a fully connected feedforward neural network

        Uses relu with no batch normalization for the hidden layers
        Linear output layer
        Clip norm of gradients
        Use mean squared error as loss function (TODO get Huber loss working)
        Weight init is left as default (random kernel, bias zeros)

        args
            model_dict (dict) : dictionary 
        """
        input_dim = model_dict['input_dim']
        layers = model_dict['layers']
        output_dim = model_dict['output_dim']
        learning_rate = model_dict['lr']

        #  setup the model
        model = keras.models.Sequential()

        #  setup the input layer
        model.add(Dense(units=layers[0],
                        input_dim=input_dim))
        model.add(Activation('relu'))

        #  setup the hidden layers
        for layer in layers[1:]:
            model.add(Dense(units=layer))
            model.add(Activation('relu'))

        #  setup the output layer
        model.add(Dense(units=output_dim))
        model.add(Activation('linear'))

        #  create optimizer
        opt = Adam(lr=learning_rate,
                   clipnorm=self.clipnorm)

        #  compile the model
        model.compile(loss='mse', optimizer=opt)

        #  we save the initial weights so that we can reset the model later
        self.initial_weights = model.get_weights()
        return model

    def copy_weights(self, parent):
        print('Copying model weights from parent into this model')
        self.model.set_weights(parent.get_weights())

    def reset_weights(self):
        """
        Resets model to initial weights
        """
        print('Resetting model weights to initial weights')
        self.model.set_weights(self.initial_weights)

    def save_model(self, path):
        """
        Saves the Keras model

        args
            path (str)
        """
        print('Saving Keras model')
        self.model.save(path)

    def load_model(self, path):
        """
        Loads a Keras model

        args
            path (str)
        """
        print('Loading Keras model')
        self.model = keras.models.load_model(path)

class KerasV(KerasFunctionApproximator):
    """
    The class for a Keras value function V(s).

    The value function approximates the future expected discounted reward
    after leaving state s.
    """
    def __init__(self, model_dict):
        super().__init__(model_dict)
        pass

    def predict(self, state):
        return self.model.predict(state)

    def improve(self, states, targets):
        return self.model.fit(x=states,
                              y=targets,
                              batch_size=self.model_dict['batch_size'],
                              epochs=self.self.model_dict['epochs'],
                              verbose=1)

class KerasQ(KerasFunctionApproximator):
    """
    The class for the action-value function Q(s,a).

    The value function approximates the future expected discounted reward
    after leaving state s, taking action a.
    """
    def __init__(self, model_dict):
        super().__init__(model_dict)

    def predict(self, state_action):
        return self.model.predict(state_action)

    def improve(self, state_actions, targets):
        return self.model.fit(x=state_actions,
                              y=targets,
                              batch_size=self.model_dict['batch_size'],
                              epochs=self.model_dict['epochs'],
                              verbose=1)
