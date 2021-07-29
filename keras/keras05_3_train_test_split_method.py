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
'''
train_size, test_size = 각각 트레인 데이터의 비율과 테스트 데이터의 비율. min 0 max 1
(한 쪽만 기입할 경우 다른 한 쪽은 자동으로 1 - 한쪽 값이 됨)
(양 쪽 다 기입하지 않을 경우 default 0.75 / 0.25)
shuffle -> 데이터를 섞어서 나눌지 안 섞어서 나눌지 정함. default True.
random_state -> 이걸 맞춰줘야 데이터가 같은 변수로 섞임. 숫자는 그냥 임의로 입력하면 됨.
'''

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