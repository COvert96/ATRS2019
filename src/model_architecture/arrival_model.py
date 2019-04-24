def build_model1():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256,
                                 activation='relu',
                                 input_shape=(x_train.shape[1],)))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_model2():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256,
                                 activation='relu',
                                 input_shape=(x_train.shape[1],)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model