from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 1 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# x_train, x_test = np.split(x, [70])  # x =  (70,) ,  (30,)
# y_train, y_test = np.split(y, [70])  # y =  (70,) ,  (30,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, shuffle=True, random_state=55)
# shuffle=True(default)

# x_train = x[:7]
# y_train = y[:7]
# x_test = x[7:] or x[-3:]
# y_test = y[7:]
print('x = ', x_train, ', ' ,x_test)
print('y = ', y_train, ', ' ,y_test)

print('x = ', x_train.shape, ', ' ,x_test.shape)
print('y = ', y_train.shape, ', ' ,y_test.shape)

# 2 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=90, batch_size=1)

# 4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# result = model.predict([11])
# print('예측값 : ', result)

y_predict = model.predict(x_test)
print('예측값 = ' ,y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 = ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse = ', rmse)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

'''
[Best Fit]
epochs=1000, batch_size=1

'''