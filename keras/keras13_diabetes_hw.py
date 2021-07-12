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

# (442, 10) (442,)

# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=57)

# 모델구성

model = Sequential()
model.add(Dense(32, input_shape=(10, ), activation='relu'))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 컴파일 및 훔련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=2, validation_split=1/19, shuffle=True)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

'''
[Best Fit]
Tree (32(+relu), 256, 64, 16(+relu), 1, )
batch_size=8, epochs=100
loss =  2189.971435546875
R2 =  0.6215347858833828
'''

# 과제 01. R2를 0.62 이상으로 올리기

# 메일에 과제의 코드를 임포트하지 말고 깃허브 주소만 적어서 보내기