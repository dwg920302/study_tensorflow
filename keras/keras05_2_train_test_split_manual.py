# keras-005 #2 [Train Test Split]

# 모든 데이터가 train / test로 나뉜 것은 아니므로, 대부분 사용자가 직접 나눠줘야 함.

from icecream import ic

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test = np.split(x, [7])
y_train, y_test = np.split(y, [7])

# x_train = x[:7]
# y_train = y[:7]
# x_test = x[7:] or x[-3:]
# y_test = y[7:]

print('x = ', x_train, ', ' ,x_test)
print('y = ', y_train, ', ' ,y_test)

# 2 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1)

# 4 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

# result = model.predict([11])
# print('예측값 : ', result)

x_predict = np.array([11])

y_predict = model.predict(x_predict)
print(x_predict, '의 예측값 = ' ,y_predict)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()