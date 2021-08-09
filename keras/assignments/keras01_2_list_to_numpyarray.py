# keras-001 #2 [list to np.array]

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1 Raw Data
x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred=[6]

# 1-1 Data Preprocessing
x = np.array(x)
y = np.array(y)
# list 형태의 x와 y를 np.array 형태로 바꾸는 과정

# 2 Create/Define Model
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3 Model Compile & Fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=5)

# 4 Evaluate or Predict
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