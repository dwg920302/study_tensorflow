import pandas as pd
import numpy as np
from icecream import ic
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime


# samsungstock2에 들어갈 7월 23일에 넣어줄 5*2줄짜리 데이터를 뽑아내기 위한 Code

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

# 7/23~ 기준으로는 y값으로 시가를 요구해서 시가를 빠는 작업을 함

y_sam = y_sam.drop('시가', axis=1)
y_sk = y_sk.drop('시가', axis=1)

ic(y_sam.shape, y_sk.shape)

# X Scaling
scalers = [QuantileTransformer() for i in range(5)]

x = scalers[4].fit_transform(x.to_numpy().reshape(x.shape[0], 1)).reshape(x.shape[0], )

pred_data = np.array([210726.0])
ic(pred_data)
pred_data = scalers[4].transform(pred_data.reshape(-1, 1)).reshape(1)
ic(pred_data)

x_train, x_test, y_sam_train, y_sam_test, y_sk_train, y_sk_test = train_test_split(
    x, y_sam, y_sk, train_size=0.7, shuffle=True, random_state=3)

# Y Scaling

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

# Model

input_1 = Input(shape=(1, ))
dense_1_1 = Dense(128, activation='relu', name='D1-1')(input_1)
# dense_1_d1 = Dropout(0.02)(dense_1_1)
dense_1_2 = Dense(512, activation='relu', name='D1-2')(dense_1_1)
dense_1_3 = Dense(1024, activation='relu', name='D1-3')(dense_1_2)
# dense_1_d2 = Dropout(1/30)(dense_1_3)
dense_1_4 = Dense(2048, activation='relu', name='D1-4')(dense_1_3)

dense_11_1 = Dense(512, activation='relu', name='DA1-1')(dense_1_4)
# dense_11_d = Dropout(0.01)(dense_11_1)
dense_11_2 = Dense(256, activation='relu', name='DA1-2')(dense_11_1)
dense_11_3 = Dense(64, activation='relu', name='DA1-3')(dense_11_2)
dense_11_4 = Dense(16, activation='relu', name='DA1-4')(dense_11_3)
output_1 = Dense(4, name='output-1')(dense_11_4)

dense_12_1 = Dense(512, activation='relu', name='DA2-1')(dense_1_4)
# dense_11_d = Dropout(0.01)(dense_11_1)
dense_12_2 = Dense(256, activation='relu', name='DA2-2')(dense_12_1)
dense_12_3 = Dense(64, activation='relu', name='DA2-3')(dense_12_2)
dense_12_4 = Dense(16, activation='relu', name='DA2-4')(dense_12_3)
output_2 = Dense(4, name='output-2')(dense_12_4)

# model = Model(inputs=input_1, outputs=output_1)

model = Model(inputs=input_1, outputs=[output_1, output_2])

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                    restore_best_weights=True)

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath_model = './samsung/_save/'
filepath_mcp = './samsung/_save/checkpoints/'
filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
modelpath = "".join([filepath_mcp, "sample_mp_",  date_time, "_", filename])
# 파일명 + 시간 + loss

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                        filepath=modelpath)

model.fit(x_train, [y_sam_train, y_sk_train], epochs=500, batch_size=4, validation_split=1/5, shuffle=True, verbose=1, callbacks=[es, mcp])
# model.fit(x_train, y_sam_train, epochs=100, batch_size=4, validation_split=1/5, shuffle=True, verbose=1, callbacks=[es, mcp])
# model.fit(x_train,  y_sk_train, epochs=100, batch_size=4, validation_split=1/10, shuffle=True, verbose=1, callbacks=[es, mcp])

model.save(filepath_model + 'sample_mp_saved_model.h5')

# model = load_model('./samsung/_save/checkpoints/sample_0723_0301_0043_0.0035.hdf5')

# loss = model.evaluate(x_test, y_sam_test)
# loss = model.evaluate(x_test, y_sk_test)
loss = model.evaluate(x_test, [y_sam_test, y_sk_test])
ic('loss = ', loss[0], loss[1], loss[2])
ic('accuracy = ', loss[3])

ic(pred_data)
predict = model.predict(pred_data)
ic(predict)

'''
[추출 샘플 모음]

(With MinMaxScaler)
epochs = 500
Epoch 00462: val_loss did not improve from 0.00359
Epoch 00462: early stopping
25/25 [==============================] - 0s 4ms/step - loss: 0.0038 - output-1_loss: 0.0015 - output-2_loss: 0.0022 - output-1_accuracy: 0.8387 - output-2_accuracy: 0.8950
ic| 'loss = ': 'loss = '
    loss[0]: 0.00379408267326653
    loss[1]: 0.0015477281995117664  <- 다른 것 보다 이걸 봐야 함
    loss[2]: 0.0022463547065854073
ic| 'accuracy = ', loss[3]: 0.8386683464050293
ic| pred_data: array([1.00004969])
ic| predict: [array([[0.5133336 , 0.54048014, 0.51698035, 0.25335014]], dtype=float32),
              array([[0.86223495, 0.89327717, 0.8608519 , 0.03713073]], dtype=float32)]

(With MaxAbsScaler)
epochs = 500
Epoch 00329: val_loss did not improve from 0.00332
Epoch 00329: early stopping
25/25 [==============================] - 0s 4ms/step - loss: 0.0034 - output-1_loss: 0.0015 - output-2_loss: 0.0020 - output-1_accuracy: 0.9142 - output-2_accuracy: 0.9577
ic| 'loss = ': 'loss = '
    loss[0]: 0.0034382983576506376
    loss[1]: 0.0014689654344692826
    loss[2]: 0.001969332806766033
ic| 'accuracy = ', loss[3]: 0.9142125248908997
ic| pred_data: array([1.00002373])
ic| predict: [array([[0.5499422 , 0.5733892 , 0.5516343 , 0.23557028]], dtype=float32),
              array([[0.88879645, 0.9153084 , 0.88215023, 0.04141053]], dtype=float32)]

(with RobustScaler)
epochs = 500
Epoch 00305: val_loss did not improve from 0.16451
Epoch 00305: early stopping
25/25 [==============================] - 0s 4ms/step - loss: 0.1399 - output-1_loss: 0.0972 - output-2_loss: 0.0426 - output-1_accuracy: 0.8105 - output-2_accuracy: 0.4840
ic| 'loss = ': 'loss = '
    loss[0]: 0.13985736668109894
    loss[1]: 0.09720882028341293
    loss[2]: 0.04264853894710541
ic| 'accuracy = ', loss[3]: 0.810499370098114
ic| pred_data: array([0.99859098])
ic| predict: [array([[1.4618939, 1.4609298, 1.4575087, 1.5442377]], dtype=float32),
              array([[ 3.1165185 ,  3.0817366 ,  3.1023796 , -0.32585227]],

(with QuantileTransformer)
epochs = 500
Epoch 00500: val_loss did not improve from 0.01124
25/25 [==============================] - 0s 4ms/step - loss: 0.0114 - output-1_loss: 0.0056 - output-2_loss: 0.0058 - output-1_accuracy: 0.8451 - output-2_accuracy: 0.4341
ic| 'loss = ': 'loss = '
    loss[0]: 0.011375080794095993
    loss[1]: 0.005555899813771248
    loss[2]: 0.005819179583340883
ic| 'accuracy = ', loss[3]: 0.8450704216957092
ic| pred_data: array([1.])
ic| predict: [array([[0.8195063, 0.8205644, 0.8192105, 0.7953807]], dtype=float32),
              array([[0.9753244 , 0.9696587 , 0.9724853 , 0.16850886]], dtype=float32)]

'''