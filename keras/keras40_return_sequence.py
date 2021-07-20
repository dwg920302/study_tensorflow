import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

# Data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11],
            [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50, 60, 70])

print(x.shape, y.shape)

x = x.reshape(x.shape[0], 3, 1)
x_predict = x_predict.reshape(1, 3, 1)

# Model

model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1), return_sequences=True))
# return sequence = 3 -> 2차원으로 된 걸 다시 3차원화
model.add(LSTM(units=20, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile, Fit

es = EarlyStopping(monitor='loss', mode='min', patience=200, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[es])

# Predict

res = model.predict(x_predict)

print(res)

'''
13/13 [==============================] - 0s 10ms/step - loss: 0.7253 - accuracy: 0.0000e+00
Epoch 00651: early stopping
[[79.773766]]
'''

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 20)                2480
_________________________________________________________________
dense (Dense)                (None, 32)                672
_________________________________________________________________
dropout (Dropout)            (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 264
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 9
=================================================================
Total params: 3,905
Trainable params: 3,905
Non-trainable params: 0
_________________________________________________________________
'''