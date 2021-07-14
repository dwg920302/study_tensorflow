# HW 2 실습 diabets

# 1. loss과 R2로 평가

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))
print(np.min(y), np.max(y))

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=38)

# 데이터 전처리(preprocess)

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

print(np.min(x_train), np.max(x_train), np.min(x_test), np.max(x_test))


input_1 = Input(shape=(10, ))
dense_1 = Dense(64)(input_1)
dense_2 = Dense(128)(dense_1)
dense_3 = Dense(64)(dense_2)
dense_4 = Dense(32)(dense_3)
output_1 = Dense(1)(dense_4)
model = Model(inputs = input_1, outputs = output_1)

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=200, verbose=0, validation_split=1/19, shuffle=True)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

# R2 구하기

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

'''
[Best Fit]
batch_size=1, epochs=200

(MinMaxScaler)
loss =  2952.057861328125
R2 =  -0.010518209210208074

(StandardScaler)
loss =  4753.3056640625
R2 =  -0.6271029495360969
'''
