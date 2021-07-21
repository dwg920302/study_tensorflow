from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
import time


datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# model = load_model('./_save/keras46_1_save_model_1.h5')
# model = load_model('./_save/keras46_1_save_model_2.h5')

model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

# model.save('./_save/keras46_1_save_model_1.h5')
# model.save_weights('./_save/keras46_1_save_weight_1.h5') # 꽝

start_time = time.time()
model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=2,
    validation_split=0.05)

model.save_weights('./_save/keras46_1_save_weight_2.h5') # 불러올 때 compile 다음에

elapsed_time = time.time() - start_time


loss = model.evaluate(x_test, y_test, batch_size=64)
print('time : ', elapsed_time)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)

'''
[Best Fit]
Epoch 1000/1000
6/6 - 0s - loss: 2719.0366 - val_loss: 2753.3696
1/1 [==============================] - 0s 14ms/step - loss: 1916.9958
time :  30.334938526153564
loss :  1916.995849609375
R^2 Score =  0.5789416486712708
'''
