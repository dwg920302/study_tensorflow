import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer

# Acc 0.98 이상 만들기


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))

print(np.unique(y_train))

# Model
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2, 2), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=10, kernel_size=(2, 2), activation='selu'))
model.add(Conv2D(filters=10, kernel_size=(2, 2), activation='elu'))
model.add(Conv2D(filters=10, kernel_size=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64))
model.add(Dense(10, activation='sigmoid'))

# preprocessing

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 스케일러 안됨;

# compile

es = EarlyStopping(monitor='val_accuracy', patience=250, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=50, epochs=500, verbose=2, validation_split=1/12, callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# with NO Scaler
# loss =  3.260441780090332 , accuracy =  0.9855999946594238