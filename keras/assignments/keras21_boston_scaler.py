from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.875, shuffle=True, random_state=38)

# 데이터 전처리(preprocess)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

print(np.min(x_train), np.max(x_train), np.min(x_test), np.max(x_test))

input_1 = Input(shape=(13, ))
dense_1 = Dense(64)(input_1)
dense_2 = Dense(256)(dense_1)
dense_3 = Dense(128)(dense_2)
dense_4 = Dense(32)(dense_3)
output_1 = Dense(1)(dense_4)
model = Model(inputs = input_1, outputs = output_1)

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=32, epochs=500, verbose=0, validation_split=1/7, shuffle=True)
# batch_size (default 32)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

# R2 구하기
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

'''
batch_size=32(default), epochs=500

(MinMaxScaler)
loss =  15.59963607788086
R2 =  0.7082643977181253

(StandardScaler)
loss =  16.309053421020508
R2 =  0.6949972965651348

(MaxAbsScaler)
loss =  15.977663040161133
R2 =  0.7011947790267149

(RobustScaler)
loss =  15.959573745727539
R2 =  0.7015330956456247

(QuantileTransformer)
loss =  18.631481170654297
R2 =  0.6515645930982312

(PowerTransformer)
loss =  14.283403396606445
R2 =  0.732879863492145
'''

# PowerTransformer가 가장 좋음, 나머지는 고만고만하고 QuantileTransformer가 가장 나쁨