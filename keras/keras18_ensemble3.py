
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# HW 2 train_size의 default 값 찾기

# HW 3 평균값과 중위값의 차이

# 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x1 = np.transpose(x1)
y1 = np.array([range(1001, 1101)])
y2 = np.array([range(1901, 2001)])
y1 = np.transpose(y1)
y2 = np.transpose(y2)

print(x1.shape, y1.shape, y2.shape)

# 모델

input_1 = Input(shape=(3, ))
dense_1 = Dense(10, activation='relu')(input_1)
dense_2 = Dense(10, activation='relu')(dense_1)
dense_3 = Dense(10, activation='relu')(dense_2)
dense_4 = Dense(10, activation='relu')(dense_3)
dense_5 = Dense(12)(dense_4)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.85, shuffle=True, random_state=24)

# 정의 순서는 Train, Test, Train, Test, ...


last_output_1 = Dense(1, name='output-1')(dense_5)

last_output_2 = Dense(1, name='output-2')(dense_5)


model = Model(inputs=input_1, outputs=[last_output_1, last_output_2])

# output은 2개의 output을 concatenate함

model.summary()

# 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train], batch_size=10, epochs=500, verbose=1, validation_split=3/17, shuffle=True)

# 평가, 예측
results = model.evaluate(x1_test, [y1_test, y2_test])
print('loss : ', results[0])
print('mae : ', results[1])

y_predict = model.predict(x1)
# print(y_predict.shape)

# r2 = r2_score(y_true=[y1, y2], y_pred=y_predict)
# print(r2)

'''
[Best Fit]
batch_size=10, epochs=500
loss :  106.760986328125
mae :  82.39360046386719
'''