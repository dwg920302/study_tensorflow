from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 데이터 구성
x = np.array([[1,2,3,4,5,6,7,8,9,10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
    # 2행 10열

print(x.shape)

x = np.transpose(x)
# 행렬 반전 : 2행 10열 -> 10행 2열

print(x.shape)

y = np.array([11,12,13,14,15,16,17,18,19,20])

print(y.shape)

# 모델 구성
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=6, epochs=6900)

# 평가 및 예측
loss = model.evaluate(x, y)
print('오차 : ', loss)

x_pred = np.array([[10, 1.3]])
# print(x_pred.shape) # (1, 2)

result = model.predict(x_pred)
print('예측 값 : ', result)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

'''
[Best Fit]
epochs=6900, batch_size=6
오차 :  0.0
예측 값 :  [[20.]]
'''

'''
[Better Fit]
epochs=6900, batch_size=6
오차 :  1.8189894306509108e-13
예측 값 :  [[20.]]
'''