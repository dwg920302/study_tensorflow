import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Model

a = np.array(range(1, 11))
size = 5

# 시계열 데이터의 X와 Y 분리시키는 함수 분석하기

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

print(dataset)

# x와 y 분리

x = dataset[:, :4]
y = dataset[:, 4]

print("x : ", x, " / ")
print("y : ", y)

x_predict = [7, 8, 9, 10]

'''
x :  [[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]  /
y :  [ 5  6  7  8  9 10]
'''



scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_pred = scaler.transform(x_predict)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=37)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(5, )))
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

