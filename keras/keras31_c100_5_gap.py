from tensorflow.keras.datasets import cifar100

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from icecream import ic
import time
from matplotlib import pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

x_train = x_train.reshape(50000, 32 * 32 * 3)/255. # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)/255. # (10000, 32, 32, 3)

# RGB값 (0-255) 을 0~1까지로 변경

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2),                          
                        padding='valid', activation='relu', 
                        input_shape=(32, 32, 3)))
model.add(Dropout(0.15))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
model.add(Dropout(0.15))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Dropout(0.15))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))

model.add(GlobalAvgPool2D())
# Flatten 역할을 하지만 원리는 전혀 다름. Flatten은 큰 판을 하나 만들어서 주지만,
# Global(A)Pool은 각 판에서 (A)에 해당하는 값을 추출. Avg면 그 판의 평균, 

model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=64, verbose=2,
    validation_split=0.05, callbacks=[es])
elapsed_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=64)
print('time : ', elapsed_time)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[1])

plt.figure(figsize=(9, 5))

plt.subplot(2, 1, 1) # 2개의 plot 중 1행 1열
plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) # 2개의 plot 중 1행 2열
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.grid()
plt.title('acc')
plt.xlabel('epoches')
plt.ylabel('acc')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()

'''
[Best Fit]
Epoch 00108: early stopping
157/157 [==============================] - 0s 3ms/step - loss: 1.9578 - accuracy: 0.4913
time :  594.8994452953339
loss[category] :  1.9577749967575073
loss[accuracy] :  0.49129998683929443

[Better Fit]
Epoch 00134: early stopping
157/157 [==============================] - 1s 3ms/step - loss: 1.9801 - accuracy: 0.4870
time :  734.6928617954254
loss[category] :  1.980082392692566
loss[accuracy] :  0.4869999885559082
'''
