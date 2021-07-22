import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.saving.save import load_model
import datetime


# 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array([range(1001, 1101)])
y1 = np.transpose(y1)

print(x1.shape, x2.shape, y1.shape)

# 모델

input_1 = Input(shape=(3, ))
dense_1_1 = Dense(10, activation='relu', name='denseA1')(input_1)
dense_1_2 = Dense(7, activation='relu', name='denseA2')(dense_1_1)
dense_1_3 = Dense(5, activation='relu', name='denseA3')(dense_1_2)
output_1 = Dense(11, name='denseA4')(dense_1_3)

input_2 = Input(shape=(3, ))
dense_2_1 = Dense(10, activation='relu')(input_2)
dense_2_2 = Dense(10, activation='relu')(dense_2_1)
dense_2_3 = Dense(10, activation='relu')(dense_2_2)
dense_2_4 = Dense(10, activation='relu')(dense_2_3)
output_2 = Dense(12)(dense_2_4)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.85, shuffle=True, random_state=24)

# 정의 순서는 Train, Test, Train, Test, ...

merge_1 = concatenate([output_1, output_2])
merge_2 = Dense(2, name='dense_beforelast')(merge_1)
last_output = Dense(1)(merge_2)
model = Model(inputs=[input_1, input_2], outputs=last_output)

# output은 2개의 output을 concatenate함

model.summary()

# 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
                    restore_best_weights=True)

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
filepath_model = './_save/'
filepath_mcp = './_save/model_checkpoint/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath_mcp, "k47_",  date_time, "_", filename])
# 파일명 + 시간 + loss

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                        filepath=modelpath)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, batch_size=10, epochs=100, verbose=1, validation_split=3/17, callbacks=[es, mcp])

model.save('./_save/keras47_3_saved_model.h5')

print('====================== 기본 출력 ========================')

# 평가, 예측
results = model.evaluate([x1_test, x2_test], y1_test)
print('loss : ', results[0])

y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y1_test, y_predict)
print('r2 : ', r2)

print('====================== load_model ========================')
model_2 = load_model('./_save/keras47_3_saved_model.h5')

results = model_2.evaluate([x1_test, x2_test], y1_test)
print('loss : ', results[0])

y_predict = model_2.predict([x1_test, x2_test])

r2 = r2_score(y1_test, y_predict)
print('r2 : ', r2)

print('====================== Model Checkpoint ========================')
# model_3 = load_model('./_save/modelcheckpoint/keras49_mcp.hdf5')

# results = model_3.evaluate([x1_test, x2_test], y1_test)
# print('loss : ', results[0])

# y_predict = model_3.predict([x1_test, x2_test])

# r2 = r2_score(y1_test, y_predict)
# print('r2 : ', r2)

'''
[Best Fit]
====================== 기본 출력 ========================
1/1 [==============================] - 0s 13ms/step - loss: 4.1718 - mae: 1.7111
loss :  4.1718220710754395
r2 :  0.9934767262543441
====================== load_model ========================
1/1 [==============================] - 0s 87ms/step - loss: 4.1718 - mae: 1.7111
loss :  4.1718220710754395
r2 :  0.9934767262543441
'''