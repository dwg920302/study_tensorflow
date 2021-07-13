import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)])
x = np.transpose(x)
print(x.shape)  # (100, 5)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape)

input_1 = Input(shape=(5, ))
dense_1 = Dense(3)(input_1)
dense_2 = Dense(4)(dense_1)
dense_3 = Dense(10)(dense_2)
output_1 = Dense(2)(dense_3)

model = Model(inputs = input_1, outputs = output_1)

model.summary()

# model = Sequential()
# model.add(Dense(3, input_shape=(5, )))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
dense (Dense)                (None, 3)                 18
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 10)                50
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22
=================================================================
Total params: 106
Trainable params: 106
Non-trainable params: 0
_________________________________________________________________
sequential과 차이는 여기는 input도 표시됨
'''
