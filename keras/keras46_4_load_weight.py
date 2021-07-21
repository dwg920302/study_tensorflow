from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from icecream import ic
import time

# 1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model

# model = load_model('./_save/keras46_1_save_model_1.h5')
# model = load_model('./_save/keras46_1_save_model_2.h5')

model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model = load_model('./_save/keras46_1_save_weight_1.h5') # 불가능

# model.save('./_save/keras46_1_save_model_1.h5')

# 3. compile fit
start_time = time.time()
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2,
    validation_split=0.05, callbacks=[es])

model.load_weights('./_save/keras46_1_save_weight_1.h5')

elapsed_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=64)
print('time : ', elapsed_time)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)

'''
[Best Fit]
'''
