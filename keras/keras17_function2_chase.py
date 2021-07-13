from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.metrics import r2_score

import numpy as np

x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred=[6]

# 1-데이터 준비
x = np.array(x)
y = np.array(y)

# 2-모델 구성

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(1))

# 2-1-모델 구성(함수형)

input_1 = Input(shape=(1, ))
dense_1 = Dense(2)(input_1)
dense_2 = Dense(4)(dense_1)
output_1 = Dense(1)(dense_2)
model = Model(inputs = input_1, outputs = output_1)

model.summary()

# 3-모델 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=20000, batch_size=1)

# 4-평가 및 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = model.predict(x_pred)
print('6의 예측 값 : ', y_pred)

y_pred = model.predict(x)
r2 = r2_score(y, y_pred)
print('r2 = ', r2)

'''
[Best Fit]
epochs=10000, batch_size=5
loss :  0.37999993562698364
예측 값 :  [[1.1999995]
 [2.0999992]
 [2.9999988]
 [3.8999984]
 [4.7999983]]
r2 =  0.8100000143043345
'''