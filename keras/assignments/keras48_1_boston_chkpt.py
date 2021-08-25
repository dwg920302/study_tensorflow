from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=92)

# 데이터 전처리(preprocess)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(13, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

# 컴파일 및 훈련
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/model_checkpoint/keras48_1_boston.hdf5')
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=1/7, shuffle=True, callbacks=[cp])

model.save('./_save/keras48_1_boston.h5')
# batch_size (default 32)

# 평가(evaluate)
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
print(y_predict)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)

'''
Epoch 100/100
12/12 [==============================] - 0s 32ms/step - loss: 20.4326 - val_loss: 12.2470
2/2 [==============================] - 0s 7ms/step - loss: 19.3002
loss =  19.300203323364258
R^2 Score =  0.7068439103275537
'''