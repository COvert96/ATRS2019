
# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.cross_validation import train_test_split
import keras
from sklearn.metrics import r2_score



flight_departures = pd.read_csv('Departing Data.csv')
flight_departures.drop(['BlockDepTime','BlockArrTime','NewMsgTime', 'AircraftType', 'BlockArrDate', 'MsgDate', 'NewMsgDate', 'BlockDepDate'],axis=1,inplace=True)



mean = flight_departures.mean(axis=0)
meanE = flight_departures['Error'].mean(axis=0)
stdE = flight_departures['Error'].std(axis=0)
flight_depdata = flight_departures - mean
std = flight_departures.std(axis=0)
flight_depdata = flight_departures / std

X = flight_depdata.drop('Error',axis=1)
y = flight_depdata['Error']
x = X.as_matrix()
y = y.as_matrix()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



flight_depdata.shape



flight_depdata.head()



x_val = x_train[:20000]
partial_x_train = x_train[20000:]

y_val = y_train[:20000]
partial_y_train = y_train[20000:]



def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128,
                           input_shape=(x_train.shape[1],)))
    model.add(keras.layers.PReLU(alpha_initializer='zeros'))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.PReLU(alpha_initializer='zeros'))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.PReLU(alpha_initializer='zeros'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "Model_Logs/model_dep_nonregular.h5",
        monitor = 'val_loss',
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_mean_absolute_error',
        patience = 7,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
    )
#     keras.callbacks.TensorBoard(
#         log_dir="./Model_Logs/Graphs/Departure",
#         histogram_freq=1,
#     )
]

model = build_model()
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=128,
                    callbacks=callbacks)



val_mae = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()



dep_model = build_model()
dep_model.fit(x_train, y_train,
         epochs=40, batch_size=128, verbose=0)



results = dep_model.evaluate(x_test, y_test)
predictions = dep_model.predict(x_test)
r2 = r2_score(y_test,predictions)



print(r2)



predicted_df = pd.DataFrame(predictions,columns=['Predicted Error'])



unstand_predict = (predicted_df * stdE) + meanE
unstand_predict.head()



test_df = pd.DataFrame(y_test,columns=['True Error'])
unstand_test = (test_df * stdE) + meanE
unstand_test.head()



# plt.plot(range(len(predicted_df)), unstand_predict['Predicted Error'], 'bo', label='Predictions')
# plt.plot(range(len(test_df)), unstand_test['True Error'], 'ro', label='True Error')
# plt.title('Predictions with True values')
# plt.xlabel('Index')
# plt.ylabel('Error')

# plt.show()

