# Machine Learning #2 [machine_learning model with iris_dataset]

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
model.fit(x, y, epochs=50, batch_size=1)

y_pred = model.predict(x)

y_pred = np.round(y_pred)

print(x, '의 예측 결과 : ', y_pred)

res = model.evaluate(x, y)
print('loss : ', res[0])
print('accuracy : ', res[1])

acc = accuracy_score(y, y_pred)
print('accuracy_score : ', acc)