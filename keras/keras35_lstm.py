import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

# Data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)

x = x.reshape(4, 3, 1)

# Model

model = Sequential()
model.add(LSTM(units=30, activation='relu', input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile, Fit

es = EarlyStopping(monitor='loss', mode='min', patience=500, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=2500, batch_size=1, callbacks=[es])

# Predict

es = EarlyStopping(monitor='loss', mode='min', patience=250)

x_input = np.array([5,6,7]).reshape(1, 3, 1)
res = model.predict(x_input)

# res = model.predict([[[5],[6],[7]]])

print(res)

'''
Epoch 1809/2500
4/4 [==============================] - 0s 7ms/step - loss: 1.8800e-05 - accuracy: 0.0000e+00
Epoch 01809: early stopping
[[7.998254]]
'''

'''
Model: "sequential" (Lstm(10))
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 601
Trainable params: 601
Non-trainable params: 0

Model: "sequential" (Lstm(20))
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 20)                1760
_________________________________________________________________
dense (Dense)                (None, 10)                210
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,981
Trainable params: 1,981
Non-trainable params: 0
_________________________________________________________________

_________________________________________________________________

Model: "sequential" (Lstm(30))
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 30)                3840
_________________________________________________________________
dense (Dense)                (None, 10)                310
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 4,161
Trainable params: 4,161
Non-trainable params: 0
_________________________________________________________________

total_params = 4 * num_units(num_units + input_dim + 1)

'''