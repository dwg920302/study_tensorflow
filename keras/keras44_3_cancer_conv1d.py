from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np


datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=92)

# 데이터 전처리(preprocess)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

model = Sequential()
model.add(Conv1D(32, kernel_size=2, padding='same', activation='relu', input_shape=(30, 1)))
model.add(Conv1D(64, kernel_size=4, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=1/7, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
Epoch 100/100
14/14 [==============================] - 0s 6ms/step - loss: 0.0039 - accuracy: 0.9955 - val_loss: 0.0249 - val_accuracy: 0.9718
3/3 [==============================] - 0s 11ms/step - loss: 0.0114 - accuracy: 0.9861
loss =  0.011362167075276375
accuracy =  0.9861111044883728
'''