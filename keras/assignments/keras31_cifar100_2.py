import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer
import time

# 어떻게 모듈 이름이 cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)/255. # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)/255. # (10000, 32, 32, 3)

# RGB값 (0-255) 을 0~1까지로 변경

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# preprocessing

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)



print(np.unique(y_train))

# Model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=50, kernel_size=(4, 4), activation='selu'))
model.add(Conv2D(filters=50, kernel_size=(4, 4), activation='elu'))
model.add(Conv2D(filters=100, kernel_size=(4, 4), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256))
model.add(Dense(100, activation='sigmoid'))


# compile

es = EarlyStopping(monitor='val_accuracy', patience=50, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

hist = model.fit(x_train, y_train, batch_size=500, validation_batch_size=50, epochs=500, verbose=2, validation_split=1/100, callbacks=[es])

end_time = time.time() - start_time

# evaluate

loss = model.evaluate(x_test, y_test)

print('Elapsed Time = ', end_time)
print('loss = ', loss[0], ', accuracy = ', loss[1])

# with NO Scaler
# valid_split 1/50 (1000)
# loss =  8.752704620361328 , accuracy =  0.3018999993801117