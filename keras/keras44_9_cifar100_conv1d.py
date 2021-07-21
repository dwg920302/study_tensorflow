from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
model.add(Conv1D(64, kernel_size=2, padding='same', activation='relu', input_shape=(1024, 3)))
model.add(Dropout(0.1))
model.add(Conv1D(128, kernel_size=4, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1024, epochs=20, verbose=1, validation_split=1/10, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
Epoch 20/20
44/44 [==============================] - 8s 180ms/step - loss: 0.0050 - accuracy: 0.5927 - val_loss: 0.0089 - val_accuracy: 0.2784
313/313 [==============================] - 1s 4ms/step - loss: 0.0088 - accuracy: 0.2873
loss =  0.008832409977912903
accuracy =  0.2872999906539917
'''