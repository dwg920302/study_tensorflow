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

# 1번째 거와 다르게, 트리를 1 -> 1로 만들어서 한번 실행하려면 sk와 samsung을 따로 뽑아야 함.

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
scalers = [MaxAbsScaler() for i in range(5)]

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
dense_1 = Dense(128, activation='relu', name='D1-1')(input_1)
dense_2 = Dense(512, activation='relu', name='D1-2')(dense_1)
dense_3 = Dense(1024, activation='relu', name='D1-3')(dense_2)
dense_4 = Dense(2048, activation='relu', name='D1-4')(dense_3)
dense_5= Dense(512, activation='relu', name='DA1-1')(dense_4)
dense_6 = Dense(256, activation='relu', name='DA1-2')(dense_5)
dense_7 = Dense(64, activation='relu', name='DA1-3')(dense_6)
dense_8 = Dense(16, activation='relu', name='DA1-4')(dense_7)
output_1 = Dense(4, name='output-1')(dense_8)

model = Model(inputs=input_1, outputs=output_1)

# model = load_model('./samsung/_save/checkpoints/sample_mp_0723_1715_0800_0.0017.hdf5')

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=200, mode='auto', verbose=1,
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

model.save(filepath_model + 'sample_mp_saved_model_v.h5')

# model.fit(x_train, y_sam_train, epochs=1000, batch_size=4, validation_split=1/7, shuffle=True, verbose=1, callbacks=[mcp])
model.fit(x_train, y_sk_train, epochs=250, batch_size=4, validation_split=1/7, shuffle=True, verbose=1, callbacks=[es, mcp])

model.save(filepath_model + 'sample_mp_saved_model_i.h5')

# res = model.evaluate(x_test, y_sam_test)
res = model.evaluate(x_test, y_sk_test)
ic('loss = ', res[0])
ic('accuracy = ', res[1])

ic(pred_data)
predict = model.predict(pred_data)
ic(predict)

'''
[추출 샘플 모음]
(With MaxAbsScaler)
epoch 1000

Epoch 01000: val_loss did not improve from 0.00174
25/25 [==============================] - 0s 3ms/step - loss: 0.0015 - accuracy: 0.9078
ic| 'loss = ', res[0]: 0.0014613786479458213
ic| 'accuracy = ', res[1]: 0.9078105092048645
ic| pred_data: array([1.00002373])
ic| predict: array([[0.54720914, 0.5710415 , 0.54856926, 0.23799258]], dtype=float32)
'''