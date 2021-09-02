import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)


from tensorflow.keras.datasets import cifar100

from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

scaler = StandardScaler()
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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2),                          
                        padding='valid', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())

model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))                   
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), activation='relu'))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))

model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2,
    validation_split=0.25, callbacks=[es])
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
loss[category] :  6.727427959442139
loss[accuracy] :  0.3682999908924103
loss[category] :  7.6312689781188965
loss[accuracy] :  0.37220001220703125
loss[category] :  6.800853729248047
loss[accuracy] :  0.3847000002861023
loss[category] :  7.04472017288208
loss[accuracy] :  0.3855000138282776
loss[category] :  6.220393180847168
loss[accuracy] :  0.4025000035762787
loss[category] :  4.9466118812561035
loss[accuracy] :  0.4169999957084656
'''

# loss[category] :  2.8825137615203857
# loss[accuracy] :  0.43130001425743103