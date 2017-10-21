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
        loss_function = model_dict['loss_function']
        optimizer = model_dict['optimizer']

        #  setup the model & input layer
        model = Sequential()
        model.add(Dense(nodes=layers[0],
                        input_dim=input_dim,
                        init='uniform'))
        model.add(Activation('relu'))

        #  setup the hidden layers
        for layer in layers[1:]:
            model.add(Dense(nodes=layer, init='uniform'))
            model.add(Activation('relu'))

        #  setup the output layer
        model.add(Dense(output_dim, init='uniform'))
        model.add(Activation('linear'))

        model.compile(loss=loss_function, optimizer=optimizer)
        return model

class Keras_V(KerasFunctionApproximator):
    """
    The class for a value function V(s)

    The value function represents the future expected discounted reward
    after leaving state s
    """
    def __init__(self, model_dict):

    def predict(self, state):
        return self.model.predict(state)

    def improve(self, states,
                      targets):
        history = self.model.fit(x=states, y=targets)
        return history

class Keras_Q(KerasFunctionApproximator):
    """
    The class for the action-value function Q(s,a)

    The value function represents the future expected discounted reward
    after leaving state s, taking action a
    """

    def predict(self, state_action):
        return self.model.predict(state_action)

    def improve(self, state_actions,
                      targets):
        history = self.model.fit(x=states, y=targets)
        return history

if __name__ == 'main':

    Q = Keras_Q(model_dict={model_type:'feedforward',
                            input_dim:10,
                            output_dim:1,
                            layers:[100, 100, 100],
                            optimizer:Adam()}
                )
