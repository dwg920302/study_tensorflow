from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalAvgPool1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

# 데이터 전처리(preprocess)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], 30, 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], 30, 1)

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=1,                   
                        padding='same', 
                        input_shape=(30, 1)))
model.add(Conv1D(32, 1, padding='same'))
# model.add(Dropout(0.1))               
# model.add(MaxPool1D())
# model.add(GlobalAvgPool1D())
model.add(Dense(1, activation='linear'))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, validation_split=1/7, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
[Best Fit]
loss =  0.18647967278957367
accuracy =  0.7208333015441895
'''