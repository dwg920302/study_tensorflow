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
from sklearn.metrics import r2_score, mean_squared_error

x_data = np.array(range(1, 101))
x_pre_data = np.array(range(96, 107))  # 106 X 107 O
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

dataset= split_x(x_pre_data, size)

print(dataset)

x_pred = dataset[:, :(size-1)]
y_exp = dataset[:, (size-1)]

print("x : ", x_pred, " / ")
print("y : ", y_exp)

# scaling

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(x.shape[0], x.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=37)

model = Sequential()
model.add(LSTM(units=16, activation='relu', input_shape=(5, 1)))
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
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
y_pred = model.predict(x_test)
print('예측값 = ', y_pred)
print(y_test.shape, y_pred.shape)
r2 = r2_score(y_test, y_pred)
print('R^2 Score = ', r2)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print('rmse = ', rmse)


'''
Epoch 10/10
75/75 [==============================] - 1s 8ms/step - loss: 17.2974 - val_loss: 12.2191
1/1 [==============================] - 0s 125ms/step - loss: 18.7226
loss =  18.722620010375977
희망값 =  [101 102 103 104 105 106] 
예측값 =  [[52.119152]
 [56.77216 ]
 [95.5468  ]
 [84.82903 ]
 [47.766968]
 [86.12246 ]
 [14.795696]
 [34.97699 ]
 [58.71982 ]
 [33.050987]
 [69.2213  ]
 [73.78563 ]
 [30.308079]
 [17.071499]
 [79.78814 ]]
(15,) (15, 1)
R^2 Score =  0.972906953566525
rmse =  4.326964251581932
'''