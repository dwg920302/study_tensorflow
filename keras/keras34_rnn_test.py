import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

# Model

model = Sequential()
model.add(SimpleRNN(units=25, activation='relu', input_shape=(3, 1)))
# model.add(SimpleRNN(units=25, input_length=3, input_dim=1, activation='relu')) # 같은 식
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model = Sequential()
model.add(SimpleRNN(units=25, activation='relu', input_shape=(3, 2)))
# model.add(SimpleRNN(units=25, input_length=3, input_dim=2, activation='relu')) # 같은 식
model.add(Dense(10))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 25)                675
_________________________________________________________________
dense (Dense)                (None, 10)                260
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 946
Trainable params: 946
Non-trainable params: 0
_________________________________________________________________
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn_1 (SimpleRNN)     (None, 25)                700
_________________________________________________________________
dense_2 (Dense)              (None, 10)                260
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 971
Trainable params: 971
Non-trainable params: 0
_________________________________________________________________

Total params = recurrent_weights + input_weights + biases
= (num_units*num_units)+(num_features*num_units) + (1*num_units)
= num_units(num_units + num_features + 1)
= num_units(num_units + input_dim + 1)

'''