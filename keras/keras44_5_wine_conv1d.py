from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1/5, shuffle=True, random_state=92)

# 데이터 전처리(preprocess)

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

model = Sequential()
model.add(Conv1D(32, kernel_size=2, padding='same', activation='relu', input_shape=(11, 1)))
model.add(Dropout(0.1))
model.add(Conv1D(128, kernel_size=4, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(1/6))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(1/6))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1, validation_split=1/16, shuffle=True)
# ValueError: Input 0 of layer dense is incompatible with the layer:
# expected axis -1 of input shape to have value 832 but received input with shape (None, 256)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
Epoch 250/250
115/115 [==============================] - 0s 4ms/step - loss: 0.0046 - accuracy: 0.9814 - val_loss: 0.0948 - val_accuracy: 0.6417
29/29 [==============================] - 0s 2ms/step - loss: 0.0866 - accuracy: 0.6746
loss =  0.08662088215351105
accuracy =  0.674646377563476
'''