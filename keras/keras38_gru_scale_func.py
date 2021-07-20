import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Dropout, Input
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

input_1 = Input(shape=(3, 1))
dense_1 = GRU(units=30, activation='relu')(input_1)
dense_2 = Dense(32, activation='relu')(dense_1)
dense_3 = Dense(8, activation='relu')(dense_2)
output_1 = Dense(1)(dense_3)

model = Model(inputs = input_1, outputs = output_1)

model.summary()

# Compile, Fit

# es = EarlyStopping(monitor='loss', mode='min', patience=200, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=1000, batch_size=1)

# model.fit(x, y, epochs=1000, batch_size=1, callbacks=[es])

# Predict

res = model.predict(x_predict)

# res = model.predict([[[5],[6],[7]]])

print(res)

# 결과값이 80에 근접하게 만들기

'''
Epoch 1000/1000
13/13 [==============================] - 0s 6ms/step - loss: 0.1126 - accuracy: 0.0000e+00
[[81.39979]]
'''


