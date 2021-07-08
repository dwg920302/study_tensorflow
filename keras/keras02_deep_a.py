from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=8000, batch_size=5)

# 4 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('예측값 : ', result)

'''
[Best Fit]
epochs=8000, batch_size=5
loss :  0.37999987602233887
예측값 :  [[5.7]]
'''