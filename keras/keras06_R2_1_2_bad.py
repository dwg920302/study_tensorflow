# HW 1
# 1 R2를 음수가 아닌 0.5 이하로 만들기 (나쁘게)
# 2 데이터 건드리지 않기
# 3 레이어는 Input output 포함 6개 이상
# 4 배치 사이즈 = 1(exact) / # 5 epo = 100 이상
# 6 Hidden Layer의 Node는 10개 이상 1000개 이하
# 7 train 비율 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


# 1 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=55)

print('x = ', x_train, ', ' ,x_test)
print('y = ', y_train, ', ' ,y_test)

print('x = ', x_train.shape, ', ' ,x_test.shape)
print('y = ', y_train.shape, ', ' ,y_test.shape)

# 2 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(25))
model.add(Dense(1000))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='ftrl', run_eagerly=False)

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4 평가, 예측
# loss = model.evaluate(x_test, y_test)
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# result = model.predict([11])
# print('예측값 : ', result)

y_predict = model.predict(x_test)
print('예측값 = ' ,y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 = ', r2)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

'''
[Best Fit]
epochs=1000, batch_size=1

'''