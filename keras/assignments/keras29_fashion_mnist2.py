import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 그냥 그대로 넣으면 차원이 맞지 않아서 형태를 재정의함

x_train = x_train.reshape(60000, 28 * 28 * 1) / 255
x_test = x_test.reshape(10000, 28 * 28 * 1) / 255

# preprocessing

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

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
model.add(Dense(10, activation='softmax'))



# compile

es = EarlyStopping(monitor='val_accuracy', patience=50, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=500, validation_batch_size=1, epochs=100, verbose=2, validation_split=1/120, callbacks=[es])

# evaluate

print(x_test.shape, y_test.shape)

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# with NO Scaler
# loss =  1.107297658920288 , accuracy =  0.9101999998092651