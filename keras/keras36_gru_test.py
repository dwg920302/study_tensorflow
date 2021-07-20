import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.python.keras.callbacks import EarlyStopping

# Model

model = Sequential()
model.add(GRU(units=10, activation='relu', input_shape=(3, 3)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

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
(50, shape 3, 1)
gru (GRU)                    (None, 40)                7950 ( 7500 + 450 ) 
(50, shape 4, 1)
gru (GRU)                    (None, 40)                7950 ( 7500 + 450 ) 
(50, shape 3, 2)
gru (GRU)                    (None, 50)                8100 ( 7500 + 600 )
(50, shape 3, 3)
gru (GRU)                    (None, 50)                8250 ( 7500 + 750 )

3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 )

'''