# keras-003 #2 [Pyplot #2]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# 데이터 구성
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])   # (3, 10)

x = np.transpose(x) # 행렬 반전 (3, 10) -> (10, 3)

y = np.array([11,12,13,14,15,16,17,18,19,20])   # (10, )

# 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=6, epochs=6750)

# 평가 및 예측
loss = model.evaluate(x, y)
print('오차 : ', loss)

x_pred = np.array([[10, 1.3, 1]])
# print(x_pred.shape) # (1, 2)

result = model.predict(x_pred)
print('예측 값 : ', result)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

'''
[Best Fit]
epochs=6750, batch_size=6
오차 :  0.0
예측 값 :  [[20.]]
'''

'''
[Better Fit]
epochs=6600, batch_size=6
오차 :  3.6379788613018216e-13
예측 값 :  [[20.]]
'''