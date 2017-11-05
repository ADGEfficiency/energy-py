import numpy as np

from keras.layers import Dense, Activation
from keras.models import Sequential


def huber_loss(y_true, y_pred):
    """
    Implementation of the Huber loss function.
    Vlad Mnih reccomends using this (ref = 2017 Deep RL Bootcamp)

    Tour of Gotcchas when implementing DQN with Keras
    """
    clip_delta=1

    error = np.abs(y_true - y_pred)
    quad = np.minimum(error, clip_delta)
    return 0.5 * np.square(quad)+ clip_delta * (error - quad)


class KerasFunctionApproximator(object):
    """
    The base class for Keras energy_py function approximators
    """
    def __init__(self, model_dict):
        self.type = model_dict['model_type']

        if self.type == 'feedforward':
            self.model = self.build_feedforward(model_dict)

    def build_feedforward(self, model_dict):
        input_dim = model_dict['input_dim']
        layers = model_dict['layers']
        output_dim = model_dict['output_dim']

        #  setup the model & input layer
        model = Sequential()
        model.add(Dense(units=layers[0],
                        input_dim=input_dim,
                        init='uniform'))
        model.add(Activation('relu'))

        #  setup the hidden layers
        for layer in layers[1:]:
            model.add(Dense(units=layer, init='uniform'))
            model.add(Activation('relu'))

        #  setup the output layer
        model.add(Dense(output_dim, init='uniform'))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='Adam')

        self.initial_weights = model.get_weights()
        return model

    def copy_weights(self, parent):
        print('copying weights')
        self.model.set_weights(parent.get_weights())

    def reset_weights(self):
        """
        Resets model to initial weights
        """
        print('resetting weights')
        self.model.set_weights(self.initial_weights) 

    def save_model(self, path):
        """
        Saves the Keras model

        args
            path (str)
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Loads a Keras model

        args
            path (str)
        """
        self.model = keras.models.load_model(path)

class Keras_ValueFunction(KerasFunctionApproximator):
    """
    The class for a Keras value function V(s).

    The value function approximates the future expected discounted reward
    after leaving state s.
    """
    def __init__(self, model_dict):
        pass

    def predict(self, state):
        return self.model.predict(state)

    def improve(self, states, targets):
        history = self.model.fit(x=states, y=targets, verbose=0)
        return history

class Keras_ActionValueFunction(KerasFunctionApproximator):
    """
    The class for the action-value function Q(s,a).

    The value function approximates the future expected discounted reward
    after leaving state s, taking action a.
    """
    def __init__(self, input_dim):
        self.model = self.build_feedforward({'layers':[1000, 1000, 1000],
                                             'input_dim':input_dim,
                                             'output_dim':1})

    def predict(self, state_action):
        return self.model.predict(state_action)

    def improve(self, state_actions, targets):
        history = self.model.fit(x=state_actions, y=targets, verbose=0)
        return history
