from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

# 데이터 구성
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    # 3행 10열

print(x.shape)

x = np.transpose(x)
# 행렬 반전 : 3행 10열 -> 10행 3열

print(x.shape)

y = np.array([11,12,13,14,15,16,17,18,19,20])

print(y.shape)

# 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(4))
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# mae = 모든 절대(절대값) 오차의 평균

start = time.time()

model.fit(x, y, batch_size=6, epochs=1000, verbose=1)

end = time.time()

# 걸린 시간

elapsed_time = end - start

# 평가 및 예측
loss = model.evaluate(x, y)
print('Loss : ', loss)
print('Elapsed Time = ', elapsed_time)

x_pred = np.array([[10, 1.3, 1]])
# print(x_pred.shape) # (1, 2)

result = model.predict(x_pred)
print('예측 값 : ', result)

y_pred = model.predict(x)
