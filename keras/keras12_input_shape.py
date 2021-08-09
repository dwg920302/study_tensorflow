# keras-012 [input_shape]

# 2차원 이상의 데이터를 집어넣어야 할 경우 input_dim 대신 input_shape를 사용함.
# input_dim = 5는 input_shape = (5,) 와 같기도 함
# 행무시 열우선 에 따라, input_shape에 값을 넣을 땐 원래 데이터의 맨 앞만 자르면 됨.
# 예를 들어, 데이터가 (824, 5) 라면 input_shape에는 (5, )를 넣어주면 됨.

from icecream import ic

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 데이터 구성
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    # 3행 10열

ic(x.shape)

x = np.transpose(x)
# 행렬 반전 : 3행 10열 -> 10행 3열

ic(x.shape)

y = np.array([11,12,13,14,15,16,17,18,19,20])

ic(y.shape)

# 모델 구성
model = Sequential()
model.add(Dense(3, input_shape=(3, )))  # model.add (Dense(3, input_dim=3)) 과 같은 뜻
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