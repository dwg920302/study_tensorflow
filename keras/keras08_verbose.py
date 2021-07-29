# keras-008 [verbose에 따른 경과시간 체크]

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
model.compile(loss='mse', optimizer='adam')

start = time.time()

model.fit(x, y, batch_size=1, epochs=1000, verbose=1)

# batch_size = 10일 때 3.3437368869781494
#

end = time.time()

elapsed_time = end - start

# 걸린 시간

# verbose = epoch 결과를 화면에 표시할 것인가 여부
# 0 = 표시 안함
# 1 = 표시함 / 2 = 표시함(축약해서, 프로그레스 바만 없앰) / 3 이상 = epoch 시행 회수만 표시함(극축약)

# 평가 및 예측
loss = model.evaluate(x, y)
print('Loss : ', loss)
print('Elapsed Time = ', elapsed_time)

x_pred = np.array([[10, 1.3, 1]])
# print(x_pred.shape) # (1, 2)

result = model.predict(x_pred)
print('예측 값 : ', result)

y_pred = model.predict(x)

'''
[Elapsed Time 차이]
verbose = 0일 때 -> Elapsed Time =  4.471315145492554
verbose = 1일 때 -> Elapsed Time =  5.918743133544922
verbose = 2일 때 -> Elapsed Time =  4.828273296356201
verbose = 3일 때 -> Elapsed Time =  4.859086513519287
0 < 2=3 <<< 1 (걸린 시간, 걸린 시간이 낮을수록 좋음)
콘솔에 출력하는 게 시간을 잡아먹음, 특히 프로그레스 바가 시간을 엄청 잡아먹음
'''
