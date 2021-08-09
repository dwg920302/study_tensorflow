# keras-003 #3 [Pyplot #3]

from icecream import ic

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt


# 데이터 구성
x = np.array([range(10), range(21, 31), range(201, 211)])   # (3, 10)

x = np.transpose(x) # 행렬 반전 : (3, 10) -> (10, 3)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])   # (3, 10)

y = np.transpose(y) # 행렬 반전 : (3, 10) -> (10, 3)


# 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(2))
model.add(Dense(3))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=10, epochs=10000)

# 평가 및 예측
loss = model.evaluate(x, y)
ic(loss)

x_pred = np.array([[0, 21, 201]])
# print(x_pred.shape) # (1, 2)

predict = model.predict(x_pred)
ic(predict)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()


'''
[Best Fit]
epochs=24000, batch_size=10
오차 :  0.005317130126059055
예측 값 :  [[1.0000002 1.1309093 9.999996 ]]
'''