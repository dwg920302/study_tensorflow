'''
# 실습
# 1~100까지의 데이터를
1, 2, 3, 4, 5   /   6
...
95, 96, 97, 98, 99   /   100

예상 결과값 -> 101, 102, 103, 104, 105, 106
평가지표 -> RMSE, R2
'''

import numpy as np
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 107))  # 106 X 107 O
'''
predict의 예상 결과값
96, 97, 98, 99, 100  /  101
...
101, 102, 103, 104, 105  /  106
'''

size = 6

def split_x(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        arr.append(subset)
    return np.array(arr)

dataset = split_x(x_data, size)

print(dataset)

x = dataset[:, :(size-1)]
y = dataset[:, (size-1)]

print("x : ", x, " / ")
print("y : ", y)

dataset= split_x(x_predict, size)

print(dataset)

x_pred = dataset[:, :(size-1)]
y_pred = dataset[:, (size-1)]

print("x : ", x_pred, " / ")
print("y : ", y_pred)

# scaling

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(x.shape[0], x.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=37)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(5, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

es = EarlyStopping(monitor='loss', mode='min', patience=20, verbose=1)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=1/17, callbacks=[es])

# Predict
y_pred = y_pred.reshape(6, 1)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
res = model.predict(x_pred)
print('희망값 = ', y_pred, '\n예측값 = ', res)
r2 = r2_score(y_pred, res)
print('R^2 Score = ', r2)

'''
loss =  1.799087405204773
희망값 =  [[101]
 [102]
 [103]
 [104]
 [105]
 [106]]
예측값 =  [[ 99.36673]
 [100.36947]
 [101.37251]
 [102.37583]
 [103.37941]
 [104.38326]]
R^2 Score =  0.09411463103190598
'''