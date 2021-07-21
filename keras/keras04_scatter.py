# keras-004 [Scatter]

# 사실 3번과 크게 다를 게 없음

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# 1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,7,9,3,8,12])

# 2 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7, input_dim=1))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

# 4 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

# result = model.predict([11])
# print('예측값 : ', result)

y_predict = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()

'''
[Best Fit]
epochs=?, batch_size=?

'''