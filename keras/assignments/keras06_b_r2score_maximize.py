# keras-006 #2-2 [r2 score_maximize]

# R2_2에서,
# R2 Score를 0.9 이상으로, 최대로 만들어보기

from icecream import ic

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import r2_score


# Data
x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred=[6]

x = np.array(x)
y = np.array(y)

# y의 Data를 Sort하면 아주 쉽게 풀 수 있겠지만 그게 정답은 아닐 거 같음.

# Model
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(1))

# 3-모델 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=5)

# 4-평가 및 예측
loss = model.evaluate(x, y)
ic(loss)

y_pred = model.predict(x_pred)
print('예측 값 : ', y_pred)

y_pred = model.predict(x)
r2_score = r2_score(y, y_pred)
ic(r2_score)

# 0.9는 아직 못함. 뭔 짓을 해도 0.81 이상이 안 됨..

# 다른 분 이거 성공한 코드가 있던데, 난이도가 ㅎㄷㄷ함..

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

