import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer

# 어떻게 모듈 이름이 cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))

print(np.unique(y_train))

# Model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='selu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='elu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(512))
model.add(Dense(100, activation='sigmoid'))

# preprocessing

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# compile

es = EarlyStopping(monitor='val_accuracy', patience=25, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=500, epochs=100, verbose=2, validation_split=1/10, callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# with NO Scaler
# loss =  4.606860160827637 , accuracy =  0.009999999776482582