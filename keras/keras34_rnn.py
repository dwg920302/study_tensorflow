import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

# Data

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)

x = x.reshape(4, 3, 1)

# Model

model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
# model.add(SimpleRNN(units=25, input_length=3, input_dim=1, activation='relu')) # 같은 식
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile, Fit

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)

# Predict

es = EarlyStopping(monitor='loss', mode='min', patience=250)

x_input = np.array([5,6,7]).reshape(1, 3, 1)
res = model.predict(x_input)

# res = model.predict([[[5],[6],[7]]])

print(res)

'''
Epoch 5000/5000
4/4 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00
[[8.]]
'''

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 241
Trainable params: 241
Non-trainable params: 0
_________________________________________________________________
'''