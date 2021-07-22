from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 데이터 전처리(preprocess)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])).reshape(
        x_train.shape[0], x_train.shape[1] * x_train.shape[2], x_train.shape[3])
x_test = scaler.transform(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])).reshape(
        x_test.shape[0], x_test.shape[1] * x_test.shape[2], x_test.shape[3])

# 모델

model = Sequential()
model.add(LSTM(8, activation='relu', input_shape=(1024, 3)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 컴파일 및 훈련
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/model_checkpoint/keras48_8_cifar10.hdf5')
model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_split=1/10, shuffle=True)

model.save('./_save/keras48_8_cifar10.h5')
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
Epoch 10/10
44/44 [==============================] - 85s 2s/step - loss: 0.0900 - accuracy: 0.1015 - val_loss: 0.0900 - val_accuracy: 0.0950
313/313 [==============================] - 119s 379ms/step - loss: 0.0900 - accuracy: 0.1000
loss =  0.09000017493963242
accuracy =  0.10000000149011612
'''