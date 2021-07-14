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
xx = Dense(3)(input_1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output_1 = Dense(2)(xx)

# 되긴 하나 이렇게 만들면 자유롭게 만들 수 없음

model = Model(inputs = input_1, outputs = output_1)

model.summary()