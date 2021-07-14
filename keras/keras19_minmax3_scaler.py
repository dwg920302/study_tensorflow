from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
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

# (x - min) / (max - min)
# 각 column 별로 min, max를 서로 다르게 구해서 해야 함

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=38)

# Train과 Test가 둘 다 포함된 전체 데이터를 스케일링하면 훈련 및 평가(Test) 시 데이터가 과적합에 걸림

scaler = MinMaxScaler()
scaler.fit(x_train)   # train만! test는 포함시키지 않고, 이 train을 fit시킨 걸로만 test함.
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

print(np.min(x_train), np.max(x_train), np.min(x_test), np.max(x_test))


input_1 = Input(shape=(13, ))
dense_1 = Dense(128, activation='relu')(input_1)
dense_2 = Dense(64)(dense_1)
dense_3 = Dense(64)(dense_2)
dense_4 = Dense(32, activation='relu')(dense_3)
output_1 = Dense(1)(dense_4)
model = Model(inputs = input_1, outputs = output_1)

# model.summary()

# x = (x - np.min(x)) / (np.max(x) - np.min(x))

# scaler를 하고 shuffle을 해야 하는 건가?

# train test 나누기



# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=16, epochs=200, verbose=0, validation_split=1/19, shuffle=True)
# batch_size (default 32)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_pred = model.predict(x_test)
print('예측값 = ', y_pred)

# R2 구하기

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

'''
[Best Fit]
batch_size=16, epochs=200
loss =  5.174633026123047
R2 =  0.9275383976082545

[Better Fit]

'''
