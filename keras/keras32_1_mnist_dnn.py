import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer

# Data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

print(np.unique(y_train))

# Data Preprocessing

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
model = Sequential()
model.add(Dense(128, input_shape=(28 * 28, )))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile and fit

es = EarlyStopping(monitor='val_accuracy', patience=25, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=2, validation_split=1/12, callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

'''
[Best Fit]
loss =  0.10423552244901657 , accuracy =  0.9807000160217285
'''