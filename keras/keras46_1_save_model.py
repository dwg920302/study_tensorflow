from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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

model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()
model.save('./_save/keras46_1_save_model_1.h5') # Summary 까지 Save. 

# 3. compile fit
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=2,
    validation_split=0.05)

model.save('./_save/keras46_1_save_model_2.h5') # Fit 까지 Save. 이후에 이 모델을 적용하면 그대로 평가에 쓸 수 있음.

elapsed_time = time.time() - start_time

y_pred = model.predict([x_test])

loss = model.evaluate(x_test, y_test)
print("elapsed time = ", elapsed_time)

print('loss : ', loss)

r2 = r2_score(y_test, y_pred)
print('R^2 score : ', r2)

'''
[Best Fit]
elapsed time =  3.7166759967803955
loss :  [1892.9757080078125, 0.0]
R^2 score :  0.5842175653264371
'''
