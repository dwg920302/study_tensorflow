import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
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

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

# Model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(28, 28)))
model.add(Flatten())    # (N, 280)
model.add(Dense(784))   # (N, 784)
model.add(Reshape((28, 28, 1)))  # (N, 28, 28, 1)
model.add(Conv2D(64, (2 ,2)))
model.add(Dropout(0.1))
model.add(Conv2D(64, (2 ,2)))
model.add(Conv2D(64, (2 ,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile and fit

es = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=2, validation_split=1/12, callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

'''
Epoch 20/20
860/860 - 6s - loss: 0.0148 - accuracy: 0.9955 - val_loss: 0.0994 - val_accuracy: 0.9782
313/313 [==============================] - 1s 3ms/step - loss: 0.1058 - accuracy: 0.9803
loss =  0.10580766946077347 , accuracy =  0.9803000092506409
'''