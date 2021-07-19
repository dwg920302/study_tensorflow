import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

print(np.unique(y_train))

# Model
model = Sequential()
model.add(Dense(100, input_shape=(28 * 28, )))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(200))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile

es = EarlyStopping(monitor='val_accuracy', patience=50, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=2, validation_split=1/120, callbacks=[es])

# evaluate

print(x_test.shape, y_test.shape)

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# with NO Scaler
# loss =  0.4104771912097931 , accuracy =  0.9016000032424927