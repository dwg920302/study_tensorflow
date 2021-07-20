from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAvgPool1D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

# 데이터 전처리(preprocess)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = QuantileTransformer()
# x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], 4, 1)
# x_test = scaler.transform(x_test).reshape(x_test.shape[0], 4, 1)
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], 4, 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], 4, 1)

model = Sequential()
model.add(Conv1D(filters=8, kernel_size=(1, ),                   
                        padding='same', 
                        input_shape=(4, 1)))
# model.add(Conv1D(filters=8, kernel_size=1,                   
#                         padding='same', 
#                         input_shape=(4, 1)))
model.add(Conv1D(filters=16, kernel_size=(1, ), padding='same'))
# model.add(Dropout(0.1))               
# model.add(MaxPool1D())
model.add(GlobalAvgPool1D())
model.add(Dense(3, activation='linear'))

# model.summary()

print(x_train.shape, y_train.shape)

# 컴파일 및 훈련

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, validation_split=1/7, shuffle=True)
# Error here. ValueError: Shapes (None, 3) and (None, 4, 3) are incompatible

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
[Best Fit]
loss =  5.942300319671631
accuracy =  0.2631579041481018
'''