from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=92)

# 데이터 전처리(preprocess)

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

model = Sequential()
model.add(Conv1D(16, kernel_size=2, padding='same', activation='relu', input_shape=(10, 1)))
model.add(Conv1D(32, kernel_size=4, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.01))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=1/7, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)


'''
Epoch 100/100
11/11 [==============================] - 0s 6ms/step - loss: 2418.0270 - val_loss: 2020.2178
2/2 [==============================] - 0s 3ms/step - loss: 4394.4224
loss =  4394.42236328125
R^2 Score =  0.35922677896742494
'''