import pandas as pd
import numpy as np
from icecream import ic
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime


# samsungstock에 들어갈 7월 23일에 넣어줄 5*2줄짜리 데이터를 뽑아내기 위한 Code

# Data

data_sam = pd.read_csv('./samsung/data/stock_samsung.csv', encoding='euc-kr')

data_sk = pd.read_csv('./samsung/data/stock_sk.csv', encoding='euc-kr')

# Preprocessing

# (1) 2011월 1월 3일 이후의 데이터만 골라내기
data_sam = data_sam[data_sam['일자'] > '2011/01/03']
data_sk = data_sk[data_sk['일자'] > '2011/01/03']
# (3601, 16) -> (2600, 16)

# (2) 열 (5+1)개 남기기
data_sam = data_sam[['일자', '시가', '고가', '저가', '종가', '거래량']]
data_sk = data_sk[['일자', '시가', '고가', '저가', '종가', '거래량']]
# (2600, 16) -> (2600, 6)

# 종가를 구하기 위해 y를 종가, 나머지 column들을 x로 나눔(6) -> (1 : 5)

x = data_sam['일자']
y_sam = data_sam.drop('일자', axis=1)
y_sk = data_sk.drop('일자', axis=1)

ic(x.shape, y_sam.shape, y_sk.shape)

# (3) 날짜가 그냥 들어가지는 않으므로, 날짜를 그대로 수치(int)로 변환시켜서 작업함
# 연도의 경우 어차피 전부 21세기 (2000~2099) 내의 날짜들이므로 앞의 20 두 숫자는 커트해버림

def time_to_int(time_string):
    year, month, day = map(int, time_string.split('/'))
    year = year % 100
    return (year*10000) + (month*100) + (day)

x = x.map(lambda a: time_to_int(a))

# 7/22 기준으로는 y값으로 종가를 요구해서 종가를 빠는 작업을 함

y_sam = y_sam.drop('종가', axis=1)
y_sk = y_sk.drop('종가', axis=1)

ic(y_sam.shape, y_sk.shape)

# X Scaling
x_scaler = MaxAbsScaler()

x = x_scaler.fit_transform(x.to_numpy().reshape(x.shape[0], 1)).reshape(x.shape[0], )

pred_data = np.array([210723.0])
ic(pred_data)
pred_data = x_scaler.transform(pred_data.reshape(-1, 1)).reshape(1)
ic(pred_data)

x_train, x_test, y_sam_train, y_sam_test, y_sk_train, y_sk_test = train_test_split(
    x, y_sam, y_sk, train_size=0.7, shuffle=True, random_state=3)

# Y Scaling
scalers = [MaxAbsScaler() for i in range(4)]

i = 0
for idx in y_sam.columns:
    ic(i, idx)
    # if idx != '일자':
    scaling_data = pd.concat([y_sam_train[idx], y_sk_train[idx]])
    # sam과 sk의 데이터를 합친 것을 fit함. 따로따로 하면 삼성의 데이터와 SK의 데이터가 서로 다른 기준으로 스케일링이 되는 단점이 있음.
    scalers[i] = scalers[i].fit(scaling_data.to_numpy().reshape(scaling_data.shape[0], 1))
    y_sam_train[idx] = scalers[i].transform(y_sam_train[idx].to_numpy().reshape(y_sam_train[idx].shape[0], 1)).reshape(y_sam_train[idx].shape[0], )
    y_sk_train[idx] = scalers[i].transform(y_sk_train[idx].to_numpy().reshape(y_sk_train[idx].shape[0], 1)).reshape(y_sk_train[idx].shape[0], )
    y_sam_test[idx] = scalers[i].transform(y_sam_test[idx].to_numpy().reshape(y_sam_test[idx].shape[0], 1)).reshape(y_sam_test[idx].shape[0], )
    y_sk_test[idx] = scalers[i].transform(y_sk_test[idx].to_numpy().reshape(y_sk_test[idx].shape[0], 1)).reshape(y_sk_test[idx].shape[0], )
    i += 1

ic(y_sam.head(), y_sk.head())


# Model

input_1 = Input(shape=(1, ))
dense_1_1 = Dense(128, activation='relu', name='D1-1')(input_1)
# dense_1_d1 = Dropout(0.02)(dense_1_1)
dense_1_2 = Dense(512, activation='relu', name='D1-2')(dense_1_1)
dense_1_3 = Dense(2048, activation='relu', name='D1-3')(dense_1_2)
# dense_1_d2 = Dropout(1/30)(dense_1_3)
dense_1_4 = Dense(1024, activation='relu', name='D1-4')(dense_1_3)

dense_11_1 = Dense(256, activation='relu', name='DA1-1')(dense_1_4)
# dense_11_d = Dropout(0.01)(dense_11_1)
dense_11_2 = Dense(64, activation='relu', name='DA1-2')(dense_11_1)
dense_11_3 = Dense(16, activation='relu', name='DA1-3')(dense_11_2)
output_1 = Dense(4, name='output-1')(dense_11_3)

dense_12_1 = Dense(256, activation='relu', name='DA2-1')(dense_1_4)
# dense_11_d = Dropout(0.01)(dense_11_1)
dense_12_2 = Dense(64, activation='relu', name='DA2-2')(dense_12_1)
dense_12_3 = Dense(16, activation='relu', name='DA2-3')(dense_12_2)
output_2 = Dense(4, name='output-2')(dense_12_3)

# model = Model(inputs=input_1, outputs=output_1)

model = Model(inputs=input_1, outputs=[output_1, output_2])

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=250, mode='auto', verbose=1,
                    restore_best_weights=True)

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath_model = './samsung/_save/'
filepath_mcp = './samsung/_save/checkpoints/'
filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
modelpath = "".join([filepath_mcp, "sample_",  date_time, "_", filename])
# 파일명 + 시간 + loss

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                        filepath=modelpath)

model.fit(x_train, [y_sam_train, y_sk_train], epochs=2500, batch_size=4, validation_split=1/5, shuffle=True, verbose=1, callbacks=[es, mcp])
# model.fit(x_train, y_sam_train, epochs=100, batch_size=4, validation_split=1/5, shuffle=True, verbose=1, callbacks=[es, mcp])
# model.fit(x_train,  y_sk_train, epochs=100, batch_size=4, validation_split=1/10, shuffle=True, verbose=1, callbacks=[es, mcp])

model.save(filepath_model + 'samplemaker_saved_model.h5')

# model = load_model('./samsung/_save/checkpoints/sample_0723_0301_0043_0.0035.hdf5')

# loss = model.evaluate(x_test, y_sam_test)
# loss = model.evaluate(x_test, y_sk_test)
loss = model.evaluate(x_test, [y_sam_test, y_sk_test])
ic('loss = ', loss[0])
ic('accuracy = ', loss[1])


# ic(pred_data)
# predict = model.predict([210723])
predict = model.predict(pred_data)
ic(predict)
# ic(y_sam_train.sort_values(by='시가', ascending=True).head())

'''
[추출 샘플 모음]
epochs = 100

[MaxAbsScaler]
Epoch 00201: val_loss did not improve from 0.00336
Epoch 00201: early stopping
25/25 [==============================] - 0s 4ms/step - loss: 0.0036 - output-1_loss: 0.0015 - output-2_loss: 0.0021 - output-1_accuracy: 0.8732 - output-2_accuracy: 0.9680
ic| 'loss = ', loss[0]: 0.0035684346221387386
ic| 'accuracy = ', loss[1]: 0.0015004368033260107
ic| predict: [array([[0.5533704 , 0.56197315, 0.58667237, 0.2527865 ]], dtype=float32),
              array([[0.8861555 , 0.89020187, 0.9189478 , 0.04637998]], dtype=float32)]

Epoch 00640: early stopping
25/25 [==============================] - 0s 4ms/step - loss: 0.0035 - output-1_loss: 0.0015 - output-2_loss: 0.0020 - output-1_accuracy: 0.8732 - output-2_accuracy: 0.9680
ic| 'loss = ', loss[0]: 0.0035044869873672724
ic| 'accuracy = ', loss[1]: 0.001480383099988103
ic| predict: [array([[0.5529626 , 0.5538426 , 0.57911247, 0.25004786]], dtype=float32),
              array([[0.87048876, 0.8706228 , 0.89793754, 0.05184639]], dtype=float32)]


[MinMaxScaler]
Epoch 00061: early stopping
25/25 [==============================] - 0s 2ms/step - loss: 0.0016 - accuracy: 0.8374
ic| 'loss = ', loss[0]: 0.001570933498442173
ic| 'accuracy = ', loss[1]: 0.8373879790306091
ic| pred_data: array([1.00001988])
ic| predict: array([[0.51100004, 0.51066434, 0.5362879 , 0.24076769]], dtype=float32)

[StandardScaler]
Epoch 00072: early stopping
25/25 [==============================] - 0s 2ms/step - loss: 0.1554 - accuracy: 0.8335
ic| 'loss = ', loss[0]: 0.15535251796245575
ic| 'accuracy = ', loss[1]: 0.8335467576980591
ic| pred_data: array([1.7079151])
ic| predict: array([[1.3296959, 1.3220351, 1.3611042, 1.4446223]], dtype=float32)

[RobustScaler]
Epoch 00056: early stopping
25/25 [==============================] - 0s 2ms/step - loss: 0.1004 - accuracy: 0.8169
ic| 'loss = ', loss[0]: 0.10039601475000381
ic| 'accuracy = ', loss[1]: 0.8169013857841492
ic| pred_data: array([0.99853144])
ic| predict: array([[1.3986331, 1.4109855, 1.4000795, 1.5008961]], dtype=float32)

[QuantileTransformer]
25/25 [==============================] - 0s 2ms/step - loss: 0.0059 - accuracy: 0.8015
ic| 'loss = ', loss[0]: 0.005942866671830416
ic| 'accuracy = ', loss[1]: 0.801536500453949
ic| pred_data: array([1.])
ic| predict: array([[0.88161135, 0.87103885, 0.8752393 , 0.79394424]], dtype=float32)

[PowerTransformer]
Epoch 00057: early stopping
25/25 [==============================] - 0s 2ms/step - loss: 0.0735 - accuracy: 0.9206
ic| 'loss = ', loss[0]: 0.07348287105560303
ic| 'accuracy = ', loss[1]: 0.9206146001815796
ic| pred_data: array([1.6516528])
ic| predict: array([[ 7.1454607e-04,  8.0569834e-04, -1.2592264e-03,  1.4205097e+00]],
                   dtype=float32)
'''