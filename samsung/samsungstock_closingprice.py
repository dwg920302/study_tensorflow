import pandas as pd
import numpy as np
from icecream import ic     # print 5글자 치기 귀찮아서 2글자짜리 ic를 썼습니다.
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime


# Data

data_sam = pd.read_csv('./samsung/stock_samsung.csv', encoding='euc-kr')

data_sk = pd.read_csv('./samsung/stock_sk.csv', encoding='euc-kr')

# ic(data_sam.head())
# ic(data_sam.tail())
# ic(data_sk.head())
# ic(data_sk.tail())
# ic(data_sam.shape, data_sk.shape)   # data_sam.shape: (3601, 16), data_sk.shape: (3601, 16)

'''
# Index(['일자', '시가', '고가', '저가', '종가', '종가 단순 5', '10', '20', '60', '120', '거래량', '단순 5', '20.1', '60.1', '120.1', 'Unnamed: 15']
ic| data_sam.head():            일자       시가       고가       저가       종가 종가 단순 5     10     20        60       120         거래량      단순 5      20.1      60.1     120.1  Unnamed: 15
                     0  2021/07/21  79000.0  79100.0  78500.0  78500.0   79380  79520  80205  80668.33     81820  12355296.0  12518607  13231829  15738260  17850522          NaN
                     1  2021/07/20  78500.0  79000.0  78400.0  79000.0   79580  79750  80285  80751.67  81879.17  12456646.0  12362675  13306892  15790504  17967754          NaN
                     2  2021/07/19  79100.0  79200.0  78800.0  79000.0   79740  79970  80335     80815  81943.33  13155414.0  11844036  13272727  15879645  18140440          NaN
                     3  2021/07/16  80100.0  80100.0  79500.0  79800.0   79880  80110  80380  80871.67     82030  10859399.0  11166574  13418124  15892634  18257966          NaN
                     4  2021/07/15  79800.0  80600.0  79500.0  80600.0   79800  80130  80415  80918.33  82088.33  13766279.0  13538810  13620990  16072245  18424651          NaN
ic| data_sam.tail():               일자       시가       고가       저가       종가 종가 단순 5   10   20   60  120         거래량 단순 5 20.1 60.1 120.1  Unnamed: 15
                     3596  2007/01/04  12220.0  12240.0  12060.0  12140.0                              19073200.0                               NaN
                     3597  2007/01/03  12540.0  12560.0  12220.0  12220.0                              19736500.0                               NaN
                     3598  2007/01/02  12400.0  12540.0  12320.0  12500.0                              17763250.0                               NaN
                     3599  2006/12/28  12320.0  12380.0  12200.0  12260.0                              10913300.0                               NaN
                     3600         NaN      NaN      NaN      NaN      NaN     NaN  NaN  NaN  NaN  NaN         NaN  NaN  NaN  NaN   NaN          NaN
ic| data_sk.head():            일자        시가        고가        저가        종가 종가 단순 5      10      20         60        120        거래량     단순 5     20.1     60.1    120.1  Unnamed: 15
                    0  2021/07/21  119500.0  120000.0  116500.0  117000.0  119900  120700  122925  124733.33  130116.67  2864601.0  2281568  2729150  3377037  4020181          NaN
                    1  2021/07/20  117500.0  119500.0  117500.0  118500.0  121200  121350  123275     124975   130212.5  2070074.0  2197265  2721103  3377708  4038547          NaN
                    2  2021/07/19  119000.0  120000.0  118500.0  119000.0  122100  122000  123450  125208.33     130300  2066638.0  2359065  2726585  3424114  4073547          NaN
                    3  2021/07/16  122000.0  122500.0  120500.0  121500.0  122300  122400  123600  125441.67  130433.33  2905546.0  2441177  2787017  3439794  4098507          NaN
                    4  2021/07/15  123500.0  124000.0  122500.0  123500.0  121900  122500  123750     125625  130491.67  1500981.0  2824784  2767046  3476711  4107026          NaN
ic| data_sk.tail():               일자       시가       고가       저가       종가 종가 단순 5   10   20   60  120        거래량 단순 5 20.1 60.1 120.1  Unnamed: 15
                    3596  2007/01/04  37100.0  37550.0  36850.0  37150.0                              4410444.0                               NaN
                    3597  2007/01/03  37550.0  37700.0  36950.0  36950.0                              3827777.0                               NaN
                    3598  2007/01/02  36900.0  37550.0  36700.0  37300.0                              4461871.0                               NaN
                    3599  2006/12/28  36800.0  36800.0  36400.0  36450.0                              2471070.0                               NaN
                    3600         NaN      NaN      NaN      NaN      NaN     NaN  NaN  NaN  NaN  NaN        NaN  NaN  NaN  NaN   NaN          NaN
         NaN
'''
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

y_sam = data_sam['종가']
x_sam = data_sam.drop('종가', axis=1)
y_sk = data_sk['종가']
x_sk = data_sk.drop('종가', axis=1)

ic(x_sam.shape, y_sam.shape)

# (3) 날짜가 그냥 들어가지는 않으므로, 날짜를 그대로 수치(int)로 변환시켜서 작업함
# 연도의 경우 어차피 전부 21세기 (2000~2099) 내의 날짜들이므로 앞의 20 두 숫자는 커트해버림

def time_to_int(time_string):
    year, month, day = map(int, time_string.split('/'))
    year = year % 100
    return (year*10000) + (month*100) + (day)

x_sam['일자'] = x_sam['일자'].map(lambda a: time_to_int(a))
x_sk['일자'] = x_sk['일자'].map(lambda a: time_to_int(a))

ic(x_sam['일자'].head(), x_sk['일자'].head())


# Split

x_sam_train, x_sam_test, x_sk_train, x_sk_test, y_sam_train, y_sam_test, y_sk_train, y_sk_test = train_test_split(
    x_sam, x_sk, y_sam, y_sk, train_size=0.7, shuffle=True, random_state=3)

# Scaling (column 별로 다르게 적용함)

scalers = [MaxAbsScaler() for i in range(5)]

x_train = np.array([x_sam_train, x_sk_train])   # (2, 2210, 5)
x_test = np.array([x_sam_test, x_sk_test])      # (2, 390, 5)

# i = 0
# for idx in x_sam.columns:
#     ic(i, idx)
#     # if idx != '일자':
#     scaling_data = pd.concat([x_sam_train[idx], x_sk_train[idx]])
#     # sam과 sk의 데이터를 합친 것을 fit함. 따로따로 하면 삼성의 데이터와 SK의 데이터가 서로 다른 기준으로 스케일링이 되는 단점이 있음.
#     scalers[i] = scalers[i].fit(scaling_data.to_numpy().reshape(scaling_data.shape[0], 1))
#     x_sam_train[idx] = scalers[i].transform(x_sam_train[idx].to_numpy().reshape(x_sam_train[idx].shape[0], 1)).reshape(x_sam_train[idx].shape[0], )
#     x_sk_train[idx] = scalers[i].transform(x_sk_train[idx].to_numpy().reshape(x_sk_train[idx].shape[0], 1)).reshape(x_sk_train[idx].shape[0], )
#     x_sam_test[idx] = scalers[i].transform(x_sam_test[idx].to_numpy().reshape(x_sam_test[idx].shape[0], 1)).reshape(x_sam_test[idx].shape[0], )
#     x_sk_test[idx] = scalers[i].transform(x_sk_test[idx].to_numpy().reshape(x_sk_test[idx].shape[0], 1)).reshape(x_sk_test[idx].shape[0], )
#     i += 1

# # ic(x_sam_train.head(), x_sam_test.head(), x_sk_train.head(), x_sk_test.head())

# # Model (Ensemble Model, 2 to 2)

# # input_1 = Input(shape=(5, ))
# # dense_1_1 = Dense(64, activation='relu', name='D1-1')(input_1)
# # dense_1_d = Dropout(0.005)(dense_1_1)
# # dense_1_2 = Dense(128, activation='relu', name='D1-2')(dense_1_d)
# # conc_point_1 = Dense(256, name='D1-3')(dense_1_2)

# # input_2 = Input(shape=(5, ))
# # dense_2_1 = Dense(64, activation='relu', name='D2-1')(input_2)
# # dense_2_d = Dropout(0.005)(dense_2_1)
# # dense_2_2 = Dense(128, activation='relu', name='D2-2')(dense_2_d)
# # conc_point_2 = Dense(256, name='D2-3')(dense_2_2)

# # dense_a_1 = concatenate([conc_point_1, conc_point_2], name='DA-1')
# # dense_a_2 = Dense(512, activation='relu', name='DA-2')(dense_a_1)
# # dense_a_d = Dropout(0.01)(dense_a_2)
# # dense_a_3 = Dense(256, activation='relu', name='DA-3')(dense_a_d)

# # dense_a1_1 = Dense(64, activation='relu', name='DA1-1')(dense_a_3)
# # dense_a1_2 = Dense(16, activation='relu', name='DA1-2')(dense_a1_1)
# # dense_a1_d = Dropout(0.005)(dense_a1_2)
# # dense_a1_3 = Dense(4, activation='relu', name='DA1-3_last_before')(dense_a1_d)
# # output_1 = Dense(1, name='output-1')(dense_a1_3)

# # dense_a2_1 = Dense(64, activation='relu', name='DA2-1')(dense_a_3)
# # dense_a2_2 = Dense(16, activation='relu', name='DA2-2')(dense_a2_1)
# # dense_a2_d = Dropout(0.005)(dense_a2_2)
# # dense_a2_3 = Dense(4, activation='relu', name='DA2-3_last_before')(dense_a2_d)
# # output_2 = Dense(1, name='output-2')(dense_a2_3)


# input_1 = Input(shape=(5, ))
# dense_1_1 = Dense(64, activation='relu', name='D1-1')(input_1)
# dense_1_2 = Dense(128, activation='relu', name='D1-2')(dense_1_1)
# conc_point_1 = Dense(256, name='D1-3')(dense_1_2)

# input_2 = Input(shape=(5, ))
# dense_2_1 = Dense(64, activation='relu', name='D2-1')(input_2)
# dense_2_2 = Dense(128, activation='relu', name='D2-2')(dense_2_1)
# conc_point_2 = Dense(256, name='D2-3')(dense_2_2)

# dense_a_1 = concatenate([conc_point_1, conc_point_2], name='DA-1')
# dense_a_2 = Dense(512, activation='relu', name='DA-2')(dense_a_1)
# dense_a_3 = Dense(256, activation='relu', name='DA-3')(dense_a_2)

# dense_a1_1 = Dense(64, activation='relu', name='DA1-1')(dense_a_3)
# dense_a1_2 = Dense(16, activation='relu', name='DA1-2')(dense_a1_1)
# dense_a1_3 = Dense(4, activation='relu', name='DA1-3_last_before')(dense_a1_2)
# output_1 = Dense(1, name='output-1')(dense_a1_3)

# dense_a2_1 = Dense(64, activation='relu', name='DA2-1')(dense_a_3)
# dense_a2_2 = Dense(16, activation='relu', name='DA2-2')(dense_a2_1)
# dense_a2_3 = Dense(4, activation='relu', name='DA2-3_last_before')(dense_a2_2)
# output_2 = Dense(1, name='output-2')(dense_a2_3)

# model = Model(inputs=[input_1, input_2], outputs=[output_1, output_2])

# # Compile and Fit

# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1,
#                     restore_best_weights=True)

# date = datetime.datetime.now()
# date_time = date.strftime("%m%d_%H%M")
# filepath_model = './samsung/_save/'
# filepath_mcp = './samsung/_save/checkpoints/'
# filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
# modelpath = "".join([filepath_mcp, "samsung_",  date_time, "_", filename])
# # 파일명 + 시간 + loss

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=modelpath)

# model.fit([x_sam_train, x_sk_train], [y_sam_train, y_sk_train], epochs=500, batch_size=16, validation_split=1/10, shuffle=True, callbacks=[es, mcp])

# model.save(filepath_model + 'samsung_saved_model.h5')

# # Evaluate and Predict
# loss = model.evaluate([x_sam_test, x_sk_test], [y_sam_test, y_sk_test])
# ic(loss)

# # 날짜만 21년 7월 23일(금요일)이고, 그 외 데이터들은 전부 21년 7월 21일 데이터로 임의로 짜깁기한 샘플을 만들어서 테스트해봄.

# # sample_1 = np.array([210723, 79000, 79100, 78500, 12355296.0])
# # sample_2 = np.array([210723, 119500, 120000, 116500, 2864601.0])

# # sample_1 = sample_1.reshape(5, 1)
# # sample_2 = sample_2.reshape(5, 1)

# # i = 0
# # for idx in x_sam.columns:
# #     # if idx != '일자':
# #     sample_1[i] = scalers[i].transform(sample_1[i].reshape(-1, 1)).reshape(1)
# #     sample_2[i] = scalers[i].transform(sample_2[i].reshape(-1, 1)).reshape(1)
# #     i += 1

# # sample_1 = sample_1.reshape(1, 5)
# # sample_2 = sample_2.reshape(1, 5)

# # [predict_1, predict_2] = model.predict([sample_1, sample_2])
# # ic(predict_1)

# Model을 가져오고, Model에서 추출한 Sample을 가져옴.

model = load_model('./samsung/_save/checkpoints/samsung_0723_0953_0327_521976.4062.hdf5')

sample_1 = np.array([1.00000949, 0.5529626 , 0.5538426 , 0.57911247, 0.25004786]).reshape(1, 5)
sample_2 = np.array([1.00000949, 0.87048876, 0.8706228 , 0.89793754, 0.05184639]).reshape(1, 5)

[predict_1, predict_2] = model.predict([sample_1, sample_2])
ic(predict_1)


'''
# 참고 : 210721의 삼성 시가는 _이다

# 3번 중 1번 꼴로 loss가 억대에서 더이상 내려가지 않는 경우가 많이 나옴.

[with MaxAbsScaler]

[1.00000949, 0.5533704 , 0.56197315, 0.58667237, 0.2527865 ], [1.00000949, 0.8861555 , 0.89020187, 0.9189478 , 0.04637998]

(No Dropout)
Epoch 00250: val_loss did not improve from 579847.75000
25/25 [==============================] - 0s 3ms/step - loss: 1825753.8750 - output-1_loss: 460829.6562 - output-2_loss: 1364924.0000
ic| loss: [1825753.875, 460829.65625, 1364924.0]
ic| predict_1: array([[84643.75]], dtype=float32)
(3rd chkpt)
ic| predict_1: array([[82621.46]], dtype=float32)

(Some Dropout)
Epoch 00250: val_loss did not improve from 667347.31250
25/25 [==============================] - 0s 3ms/step - loss: 3277119.5000 - output-1_loss: 367755.8125 - output-2_loss: 2909363.2500
ic| loss: [3277119.5, 367755.8125, 2909363.25]
ic| predict_1: array([[82267.5]], dtype=float32) <- This

Epoch 00375: early stopping
25/25 [==============================] - 0s 3ms/step - loss: 544324.4375 - output-1_loss: 181430.7031 - output-2_loss: 362893.7188
ic| loss: [544324.4375, 181430.703125, 362893.71875]
ic| predict_1: array([[83050.58]], dtype=float32)

(best_weight)
[1.00000949, 0.5529626 , 0.5538426 , 0.57911247, 0.25004786], [1.00000949, 0.87048876, 0.8706228 , 0.89793754, 0.05184639]

'''