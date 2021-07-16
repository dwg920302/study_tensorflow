from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential() 
model.add(Conv2D(10, kernel_size=(2, 2), # (N, 10, 10, 1)
        padding='same', input_shape=(10, 10, 1))) # (N, 10, 10, 10)
# Conv2D(장수, 단위 사이즈, 크기 - >연산량 = input_shape(x-kernel+1) * (y-kernel+1) * 장수)
# padding='same' == 원본의 크기를 유지함 -> (5, 5, 10). default 'valid'
model.add(Conv2D(20, (2, 2), activation='relu')) # (N, 9, 9, 20)
model.add(Conv2D(30, (2, 2), activation='relu')) # (N, 8, 8, 30)
model.add(MaxPooling2D()) # (N, 4, 4, 30)
model.add(Conv2D(15, (2, 2), activation='relu')) # (N, 3, 3, 15)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

