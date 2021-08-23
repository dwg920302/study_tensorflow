from icecream import ic # print 대신

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17    <-
Non-trainable params: 0
_________________________________________________________________
'''

print(model.weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 1.0474089 ,  0.5243356 , -0.80676603]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[ 0.23217571,  0.29378736],
       [-0.6047634 ,  0.7332277 ],
       [ 0.72765017,  0.02703154]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[ 0.05371571],
       [-1.033557  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print(model.trainable_weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.61259353, -0.10978365,  0.3636937 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[ 0.2403158 ,  0.8303251 ],
       [ 0.11057293, -0.6972549 ],
       [-0.6838227 ,  0.519673  ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-1.0177135 ],
       [ 0.08173013]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''