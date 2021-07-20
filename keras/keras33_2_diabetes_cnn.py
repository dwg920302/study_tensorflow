from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalAvgPool1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score
import numpy as np

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

# 데이터 전처리(preprocess)

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], 10, 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], 10, 1)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=1,                   
                        padding='same', 
                        input_shape=(10, 1)))
model.add(Conv1D(32, 1, padding='same'))
# model.add(Dropout(0.1))               
# model.add(MaxPool1D())
model.add(GlobalAvgPool1D())
model.add(Dense(1))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=1/7, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)

# 현재 ISSUE = 작동은 하는데 ACCURACY가 무조건 0이 나옴
'''
Epoch 100/100
11/11 [==============================] - 0s 5ms/step - loss: 5126.3151 - val_loss: 4602.6821
2/2 [==============================] - 0s 2ms/step - loss: 3357.1699
loss =  3357.169921875
R^2 Score =  0.2626147557049694
'''