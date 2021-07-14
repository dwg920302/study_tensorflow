from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np

datasets = load_diabetes()
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

input_1 = Input(shape=(10, ))
dense_1 = Dense(64)(input_1)
dense_2 = Dense(128)(dense_1)
dense_3 = Dense(128)(dense_2)
dense_4 = Dense(32)(dense_3)
dense_4 = Dense(8)(dense_3)
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
batch_size=default 32, epochs=500

(MinMaxScaler)
loss =  3043.041748046875
R2 =  0.2938798652649476

(StandardScaler)
loss =  2942.64453125
R2 =  0.31717647342583877

(MaxAbsScaler)
loss =  2934.279052734375
R2 =  0.31911761940385774

(RobustScaler)
loss =  2943.43701171875
R2 =  0.31699262533079864

(QuantileTransformer)
loss =  2838.54638671875
R2 =  0.34133186076159794

(PowerTransformer)
loss =  2978.700439453125
R2 =  0.30880998160610107
'''

# QuantileTransformer가 가장 좋음, 나머지는 고만고만하고 MinMaxScaler가 가장 나쁨