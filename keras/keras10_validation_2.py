from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1 데이터
x = np.array(range(1, 14))
y = np.array(range(1, 14))

# train_test_split으로 데이터 나누기

x_train = np.array(range(1, 8))
y_train = np.array(range(1, 8))
x_test = np.array(range(8, 11))
y_test = np.array(range(8, 11))
x_valid = np.array(range(11, 14))
y_valid = np.array(range(11, 14))

print(x_train, x_test, x_valid)


#데이터가 한쪽으로 과적합 되는 걸 막기 위해 shuffle 뒤에 train/test/valid를 나눔.

#loss는 통상적으로 val_loss보다 잘 나옴.

# 2 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7, input_dim=1))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_valid, y_valid))

# 4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# result = model.predict([11])
# print('예측값 : ', result)

x_predict = np.array([14])

y_predict = model.predict(x_predict)
print(x_predict, '의 예측값 = ' ,y_predict)