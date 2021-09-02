import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

from tensorflow.keras.datasets import mnist

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from icecream import ic
import time
from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical


# strategy = tf.distribute.MirroredStrategy()   

# GPU 2개를 동시에 씀

# strategy = tf.distribute.MirroredStrategy(
#     # cross_device_ops=tf.distribute.HierarchicalCopyAllReduce() # 375
#     cross_device_ops=tf.distribute.ReductionToOneDevice() # 375
# )

# strategy = tf.distribute.MirroredStrategy(
#     devices=['/gpu:1']
#     # devices=['/cpu', '/gpu:0']
#     # devices=['/cpu', '/gpu:0', '/gpu:1']

#     # 단, GPU는 2개 이상 같이 쓰려면 cross_device_ops를 활성화시켜 줘야 함
#     # 안 그러면 Adam.NcclAllReduce 에러 남
# )

# strategy = tf.distribute.experimental.CentralStorageStrategy()

# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    # tf.distribute.experimental.CollectiveCommunication.AUTO
)

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# ic(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
# ic(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

# x_train = x_train.reshape(50000, 32 * 32 * 3)/255. # (50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32 * 32 * 3)/255. # (10000, 32, 32, 3)

# # RGB값 (0-255) 을 0~1까지로 변경

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (50000, 100)
# y_test = one.transform(y_test).toarray() # (10000, 100)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

with strategy.scope():

    # 2. model
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(2, 2),                          
                            padding='valid', activation='relu', 
                            input_shape=(28, 28, 1))) 
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
    model.add(Dense(10, activation='softmax'))

    # 3. comple fit // metrics 'acc'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2,
    validation_split=0.2, callbacks=[es])  # 분산 처리시에는 배치 사이즈가 큰 게 좋음
elapsed_time = time.time() - start_time

# Why 375? -> 48000 / 128 != 57

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


