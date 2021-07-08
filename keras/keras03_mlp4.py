from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 데이터 구성
x = np.array([range(10)])
    # 3행 10열

x = np.transpose(x)
# 행렬 반전 : 3행 10열 -> 10행 3열

print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])

y = np.transpose(y)

print(y.shape)

# 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(2))
model.add(Dense(3))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=30, epochs=6000)

# 평가 및 예측
loss = model.evaluate(x, y)
print('오차 : ', loss)

x_pred = np.array([[9]])
# print(x_pred.shape) # (1, 2)

result = model.predict(x_pred)
print('예측 값 : ', result)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

'''
[Best Fit]
epochs=9000, batch_size=30
오차 :  0.005317172966897488
예측 값 :  [[10.         1.5290909  1.0000007]]

'''

'''
[Better Fit]
epochs=2500, batch_size=10
오차 :  0.005317174829542637
예측 값 :  [[10.         1.5290898  1.0000018]]
epochs=6000, batch_size=30
오차 :  0.005317169241607189
예측 값 :  [[10.         1.5290909  0.9999991]]
'''