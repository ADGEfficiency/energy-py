from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import keras.backend.tensorflow_backend as KTF


def Dense_Q(input_length, device):
    devices = ['cpu:0', 'gpu:0']
    with KTF.tf.device(devices[device]):  # force tensorflow to train on GPU
        KTF.set_session(
            KTF.tf.Session(
                config=KTF.tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
                )
            )
        model = Sequential()
        model.add(Dense(units=500, input_shape=(input_length,),
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(units=250, input_shape=(input_length,),
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(units=100, input_shape=(input_length,),
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(output_dim=1, activation='linear',
                        kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=adam())
    return model
