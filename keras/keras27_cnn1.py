from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential() # (N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(5, 5, 1), padding='same')) # (N, 4, 4, 10)
# Conv2D(장수, 단위 사이즈, 크기 - >연산량 = input_shape(x-kernel+1) * (y-kernel+1) * 장수)
# padding='same' == 원본의 크기를 유지함 -> (5, 5, 10). default 'valid'
model.add(Conv2D(20, (2, 2), activation='relu')) # (N, 3, 3, 20)

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
flatten (Flatten)            (None, 160)               0
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
_________________________________________________________________
4차원을 2차원화시킴.
'''