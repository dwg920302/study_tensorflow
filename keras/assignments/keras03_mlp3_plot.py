from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 데이터 구성
x = np.array([range(10), range(21, 31), range(201, 211)])
    # 3행 10열

print(x.shape)

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
model.add(Dense(10, input_dim=3))
model.add(Dense(2))
model.add(Dense(3))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=10, epochs=10000)

# 평가 및 예측
loss = model.evaluate(x, y)
print('오차 : ', loss)

x_pred = np.array([[0, 21, 201]])
# print(x_pred.shape) # (1, 2)

result = model.predict(x_pred)
print('예측 값 : ', result)

x_plot = np.transpose(x)
y_pred = model.predict(x)

for x in x_plot:
    for i in range(3):  # 3 안에 다른 적합한 값을 찾아야 함
        plt.scatter(x, np.transpose(y)[i])
        plt.plot(x, np.transpose(y_pred)[i])
    plt.show()



'''
[Best Fit]
epochs=24000, batch_size=10
오차 :  0.005317130126059055
예측 값 :  [[1.0000002 1.1309093 9.999996 ]]
'''