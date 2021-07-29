# keras-006 #2 [r2 score]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
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

y_predict = model.predict(x)
print('예측 값 : ', y_predict)

r2 = r2_score(y, y_predict)
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
