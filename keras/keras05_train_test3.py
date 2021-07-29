# keras-005 #3 [train_test_split method]

# train과 test를 자동으로 나눠주는 메소드

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


# 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# x_train, x_test = np.split(x, [70])  # x =  (70,) ,  (30,)
# y_train, y_test = np.split(y, [70])  # y =  (70,) ,  (30,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, shuffle=True, random_state=55)
# shuffle=True(default)

print('x = ', x_train, ', ' ,x_test)
print('y = ', y_train, ', ' ,y_test)

print('x = ', x_train.shape, ', ' ,x_test.shape)
print('y = ', y_train.shape, ', ' ,y_test.shape)

# 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# result = model.predict([11])
# print('예측값 : ', result)

x_predict = np.array([100])

y_predict = model.predict(x_predict)
print(x_predict, '의 예측값 = ' ,y_predict)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

'''
[Best Fit]
epochs=1000, batch_size=1

'''