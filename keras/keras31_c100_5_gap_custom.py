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
from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, MaxPool2D, Dropout, GlobalAvgPool2D

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2),                          
                        padding='valid', activation='relu', 
                        input_shape=(32, 32, 3)))
model.add(Dropout(4/30))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
model.add(Dropout(4/30))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))    
model.add(AvgPool2D())

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Dropout(4/30))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))

model.add(GlobalAvgPool2D())

model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=64, verbose=2,
    validation_split=0.02, callbacks=[es])
elapsed_time = time.time() - start_time

# 4. evaluate

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
Epoch 00162: early stopping
157/157 [==============================] - 1s 3ms/step - loss: 1.9620 - accuracy: 0.4963
time :  917.327419757843
loss[category] :  1.961956262588501
loss[accuracy] :  0.49630001187324524

[Better Fit]
Epoch 00082: early stopping
157/157 [==============================] - 1s 3ms/step - loss: 1.9348 - accuracy: 0.4946
time :  459.5307447910309
loss[category] :  1.9347658157348633
loss[accuracy] :  0.49459999799728394
'''
