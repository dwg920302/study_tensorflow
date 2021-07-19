import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train = x_train.reshape(50000, 32 * 32 * 3) / 255
x_test = x_test.reshape(10000, 32 * 32 * 3) / 255

print(x_train.shape)

# preprocessing

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.unique(y_train))

# Model
model = Sequential()
model.add(Dense(100, input_shape=(32 * 32 * 3, )))
model.add(Dense(384, activation='relu'))
model.add(Dense(1024))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))


# compile

es = EarlyStopping(monitor='val_accuracy', patience=25, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=2, validation_split=1/1000, callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# with NO Scaler
# loss =  5.124889373779297 , accuracy =  0.5336999893188477