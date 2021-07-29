from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# Data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(16, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# Compile & Fit
start_time = time.time()
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='../_save/model_checkpoint/keras47_mcp.hdf5')

hist = model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2,
    validation_split=0.05, callbacks=[es, cp])

model.save('../_save/keras47_mcp.h5')

elapsed_time = time.time() - start_time

# Predict, Evaluate

loss = model.evaluate(x_test, y_test, batch_size=64)
print('time : ', elapsed_time)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)

'''
[Best Fit]
Epoch 500/500
6/6 - 0s - loss: 2659.5144 - val_loss: 3402.2019
1/1 [==============================] - 0s 12ms/step - loss: 1891.3756
time :  21.928235292434692
loss :  1891.3756103515625
R^2 Score =  0.5845690115048825
'''
