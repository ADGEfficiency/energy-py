from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam


def Dense_Q(input_length):
    model = Sequential()
    model.add(Dense(units=50, input_shape=(input_length,),
                    activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(units=25, input_shape=(input_length,),
                    activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_dim=1, activation='linear',
                    kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=adam())
    return model
