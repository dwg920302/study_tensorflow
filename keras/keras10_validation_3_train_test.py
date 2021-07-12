from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 1 데이터
x = np.array(range(1, 14))
y = np.array(range(1, 14))

# train_test_split으로 데이터 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, random_state=42)

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