import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)

# (442, 10) (442,)

print(dataset.feature_names)

# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(dataset.DESCR)

print(np.min(y), np.max(y))

# 데이터 마저 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=61)

# 모델구성

model = Sequential()
model.add(Dense(100, input_shape=(10, ), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
# activation은 마지막에 안 넣어줌

# 컴파일 및 훔련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=200, verbose=2, validation_split=0.25, shuffle=True)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)