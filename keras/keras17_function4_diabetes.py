import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score

dataset = load_diabetes()
x = dataset.data
y = dataset.target

# (442, 10) (442,)

# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=57)

# 모델구성

# model = Sequential()
# model.add(Dense(32, input_shape=(10, ), activation='relu'))
# model.add(Dense(256))
# model.add(Dense(64))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

input_1 = Input(shape=(10, ))
dense_1 = Dense(32)(input_1)
dense_2 = Dense(256)(dense_1)
dense_3 = Dense(64)(dense_2)
dense_4 = Dense(16)(dense_3)
output_1 = Dense(1)(dense_4)
model = Model(inputs = input_1, outputs = output_1) 


# 컴파일 및 훔련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=2, validation_split=1/19, shuffle=True)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)