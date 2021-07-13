# 데이터 array를 np로 바꿔주기

# 완성한 뒤, 출력 결과(loss 값, 예측 값) 를 캡쳐할 것
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred=[6]

# 1-데이터 준비
x = np.array(x)
y = np.array(y)

# 2-모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3-모델 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=5)

# 4-평가 및 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('예측 값 : ', result)

'''
[Best Fit]
epochs=10000, batch_size=5
loss :  0.3800000250339508
예측 값 :  [[5.7]]
'''