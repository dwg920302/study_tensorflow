# keras-010 #2 [Validation #2]

# 1번과 다른 점은 array를 range로 정의 한 것 (생긴 건 같음)

from icecream import ic

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 데이터
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


#loss는 통상적으로 val_loss보다 잘 나옴.

# 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7, input_dim=1))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_valid, y_valid))

# 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

# result = model.predict([11])
# print('예측값 : ', result)

x_predict = np.array([14])

y_predict = model.predict(x_predict)
print(x_predict, '의 예측값 = ' ,y_predict)