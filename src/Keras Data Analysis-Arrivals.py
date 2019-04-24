
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.cross_validation import train_test_split
import keras
from sklearn.metrics import r2_score


flight_arrivals = pd.read_csv('Arriving Data.csv')
flight_arrivals.drop('NewMsgTime',axis=1,inplace=True)
flight_arrivals.head()


flight_arrivals.drop(['AircraftType','BlockArrTime','DepCountry','BlockArrDate','MsgDate', 'NewMsgDate', 'BlockDepDate'],axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()


# flight_arrivals['AircraftType'] = number.fit_transform(flight_arrivals['AircraftType'])

flight_arrivals.shape


mean = flight_arrivals.mean(axis=0)
meanE = flight_arrivals['Error'].mean(axis=0)
stdE = flight_arrivals['Error'].std(axis=0)
flight_arrdata = flight_arrivals - mean
std = flight_arrivals.std(axis=0)
flight_arrdata = flight_arrdata / std


flight_arrdata.shape


X = flight_arrdata.drop('Error',axis=1)
y = flight_arrdata['Error']

x = X.as_matrix()
y = y.as_matrix()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


x_train.shape



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


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "Model_Logs/Model_Arr_NonReg/model1_arr_nonregular.h5",
        monitor = 'val_loss',
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_mean_absolute_error',
        patience = 15,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
    )
#     keras.callbacks.TensorBoard(
#         log_dir="./Model_Logs/Graphs/Arrivals",
#         histogram_freq=1,
#     )
]

model1 = build_model1()

history = model1.fit(x_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)


#keras.utils.plot_model(model, show_shapes=True, to_file='model.png')


val_mae = history.history['val_mean_absolute_error']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


mae_history = history.history['val_mean_absolute_error']

plt.plot(range(1, len(mae_history) + 1), mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

plt.show()


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "Model_Logs/Model_Arr_NonReg/model2_arr_nonregular.h5",
        monitor = 'val_loss',
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor = 'val_mean_absolute_error',
        patience = 15,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
    )
#     keras.callbacks.TensorBoard(
#         log_dir="./Model_Logs/Graphs/Arrivals",
#         histogram_freq=1,
#     )
]

model2 = build_model2()

history = model2.fit(x_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_split=0.2,
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
plt.legend()

plt.show()


mae_history = history.history['val_mean_absolute_error']

plt.plot(range(1, len(mae_history) + 1), mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

plt.show()


arr_model1 = build_model1()

arr_model1.fit(x_train, y_train,
         epochs=52, batch_size=128)


results1 = arr_model1.evaluate(x_test, y_test)
preds1 = arr_model1.predict(x_test)
r2_1 = r2_score(y_test,preds1)


results1[0]

arr_model2 = build_model2()

arr_model2.fit(x_train, y_train,
         epochs=44, batch_size=128)


# In[28]:


results2 = arr_model1.evaluate(x_test, y_test)
preds2 = arr_model1.predict(x_test)
r2_2 = r2_score(y_test,preds2)


results2[0]


final_preds = 0.5*(preds1+preds2)

r2 = r2_score(y_test,final_preds)


predicted_df = pd.DataFrame(final_preds,columns=['Predicted Error'])


# stand_feat = pd.DataFrame(x_test,columns=flight_data.columns[:-1])


unstand_predict = (predicted_df * stdE) + meanE
unstand_predict.head()


test_df = pd.DataFrame(y_test,columns=['True Error'])
unstand_test = (test_df * stdE) + meanE
unstand_test.head()

print(r2)

