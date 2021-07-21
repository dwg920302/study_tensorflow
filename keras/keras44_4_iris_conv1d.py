from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=13/16, shuffle=True, random_state=92)

# 데이터 전처리(preprocess)

print(type(x_train))

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# validation, predict의 경우에도 fit을 진행하지 않고 transform만 함

model = Sequential()
model.add(Conv1D(8, kernel_size=2, padding='same', activation='relu', input_shape=(4, 1)))
model.add(Conv1D(16, kernel_size=4, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.05))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

# model.summary()

# 컴파일 및 훈련

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=250, verbose=1, validation_split=1/13, shuffle=True)
# batch_size (default 32)

# 평가(evaluate)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
Epoch 250/250
4/4 [==============================] - 0s 9ms/step - loss: 0.0089 - accuracy: 0.9964 - val_loss: 0.0013 - val_accuracy: 1.0000
1/1 [==============================] - 0s 33ms/step - loss: 0.3143 - accuracy: 0.8966
loss =  0.3142915964126587
accuracy =  0.8965517282485962
'''