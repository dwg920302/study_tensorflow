# keras-002 #2 [잘못된 예시]

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5,6])

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=500, batch_size=5)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('예측값 : ', result)

'''
ValueError: Data cardinality is ambiguous:
  x sizes: 5
  y sizes: 6
Make sure all arrays contain the same number of samples.
-> 비교하는 x와 y의 길이가 같아야 하는데 달라서 발생하는 오류임.
'''