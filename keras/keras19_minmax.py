from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np


datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))
print(np.min(y), np.max(y))

print(x.shape)
print(y.shape)

# 데이터 전처리(preprocess)

# x의 minmax 0.0 711.0

# min이 0이 아닐 수도 있음

x = (x - np.min(x)) / (np.max(x) - np.min(x))

# model = Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(13))
# model.add(Dense(1))

input_1 = Input(shape=(13, ))
dense_1 = Dense(128)(input_1)
dense_2 = Dense(64)(dense_1)
dense_3 = Dense(64)(dense_2)
output_1 = Dense(1)(dense_3)
model = Model(inputs = input_1, outputs = output_1)

# model.summary()

# train test 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=38)

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=10, epochs=1000, verbose=0, validation_split=1/18, shuffle=True)
# batch_size (default 32)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_pred = model.predict(x_test)
print('예측값 = ', y_pred)

# R2 구하기

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)