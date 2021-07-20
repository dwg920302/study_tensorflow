import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.python.keras.callbacks import EarlyStopping

# Data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)

x = x.reshape(4, 3, 1)

# Model

model = Sequential()
model.add(GRU(units=50, activation='relu', input_shape=(3, 1)))
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
(국뽕) - > 한국인 조경현 박사가 만듦

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390 ( 300 + 90 )
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 511
Trainable params: 511
Non-trainable params: 0
_________________________________________________________________
(20)
gru (GRU)                    (None, 20)                1380 ( 1200 + 180 )
(30)
gru (GRU)                    (None, 30)                2970 ( 2700 + 270 )
(40)
gru (GRU)                    (None, 40)                5160 ( 4800 + 360 )
(50)
gru (GRU)                    (None, 40)                7950 ( 7500 + 450 ) 

total_params = 3 * num_units(num_units + input_dim + 2)

'''