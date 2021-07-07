from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)
#epochs=학습 회수, batch_size=표본의 크기(?)

# 4 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4, 5])
print('예측값 : ', result)