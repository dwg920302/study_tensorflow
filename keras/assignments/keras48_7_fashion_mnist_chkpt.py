from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 데이터 전처리(preprocess)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])).reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = scaler.transform(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])).reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

# 모델

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(28, 28)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 컴파일 및 훈련
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/model_checkpoint/keras48_7_fashionmnist.hdf5')
model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=256, epochs=25, verbose=1, validation_split=1/12, shuffle=True)

model.save('./_save/keras48_7_fashionmnist.h5')
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
Epoch 25/25
215/215 [==============================] - 15s 68ms/step - loss: 0.0187 - accuracy: 0.8717 - val_loss: 0.0197 - val_accuracy: 0.8608
313/313 [==============================] - 4s 13ms/step - loss: 0.0202 - accuracy: 0.8599
loss =  0.02017950266599655
accuracy =  0.8598999977111816
'''