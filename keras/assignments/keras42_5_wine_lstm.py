from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd

dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality'].to_numpy()
x = dataset.drop(columns='quality')

print(x, y)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=3/4, shuffle=True, random_state=92)

# 데이터 전처리(preprocess)

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(11, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=1/12, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

# 

'''
Epoch 100/100                                                                                                            cy: 0.5416
115/115 [==============================] - 3s 29ms/step - loss: 0.0766 - accuracy: 0.5960 - val_loss: 0.0842 - val_accuracy: 0.5302                                                                                                               cy: 0.5302
20/20 [==============================] - 0s 6ms/step - loss: 0.0818 - accuracy: 0.5726
loss =  0.08177541941404343
accuracy =  0.5725938081741333
'''