from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from icecream import ic
import time
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=92)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], x_test.shape[1], 1, 1)

# 2. model

model = Sequential()
model.add(Conv1D(16, kernel_size=2, padding='same', activation='relu', input_shape=(13, 1)))
model.add(Conv1D(32, kernel_size=4, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.01))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. comple fit // metrics 'acc'
model.compile(loss='mse', optimizer=Adam(lr=0.1))

es = EarlyStopping(monitor='val_loss', patience=51, mode='min', verbose=1, restore_best_weights=True)

# ReduceLRonPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

# Epoch 00013: ReduceLROnPlateau reducing learning rate to 0.05000000074505806.

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, verbose=2,
    validation_split=0.1, callbacks=[es, reduce_lr])
elapsed_time = time.time() - start_time

# 4. predict eval
y_pred = model.predict(x_test)
r2_score = r2_score(y_test, y_pred)
result = model.evaluate(x_test, y_test)
print('time : ', elapsed_time)
print('loss : ', result)
print('r2_score : ', r2_score)

# plt.figure(figsize=(9, 5))

# plt.subplot(2, 1, 1) # 2개의 plot 중 1행 1열
# plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.xlabel('epoches')
# plt.ylabel('loss')
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2) # 2개의 plot 중 1행 2열
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.grid()
# plt.title('acc')
# plt.xlabel('epoches')
# plt.ylabel('acc')
# plt.legend(['accuracy', 'val_accuracy'])

# plt.show()

# Issue ->  r2_score와 loss가 매번 제멋대로 튐