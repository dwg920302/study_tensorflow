# keras-005 [Train, Test]

# 훈련(Train(ing)) 표본과 테스트 표본은 원래 서로 다르게 잡아야 함
# Train은 훈련시킬 데이터, test는 평가할 데이터

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic


# 1 데이터

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7, input_dim=1))
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