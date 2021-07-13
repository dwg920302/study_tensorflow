import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)])
x = np.transpose(x)
print(x.shape)  # (100, 5)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape)

model = Sequential()
model.add(Dense(3, input_shape=(5, )))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 18   #   input*dense + dense(bias)  3*5 + 3 -> (input+1) * dense
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
'''