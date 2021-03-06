from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAvgPool2D, MaxPool2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from icecream import ic
import time

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

# train 내부 폴더 -> y의 라벨로 자동 지정됨
# x, y 자동 생성
data = datagen.flow_from_directory(
    '../_data/rps',
    target_size=(128, 128),
    batch_size=5000,
    class_mode='categorical',
    shuffle=True
)
# 분류가 2종류(binary)가 아니라 3종류기 때문에 binary가 아니라 categorical로 해야 함

# Found 3309 images belonging to 2 classes.

ic(data)

start_time = time.time()

# data_x = data[0][0] # 이미지파일 같음
# data_y = data[0][1] # 인덱스 같음 (남=0 여=1)

# np.save('/_save/_npy/k59_x_data_rps.npy', arr=data_x)
# np.save('/_save/_npy/k59_y_data_rps.npy', arr=data_y)

data_x = np.load('../_save/_npy/k59_x_data_rps.npy')
data_y = np.load('../_save/_npy/k59_y_data_rps.npy')

elapsed_time_get_data = time.time() - start_time

ic(elapsed_time_get_data)   
ic(data_x.shape, data_y.shape)

# ic| data_x.shape: (2520, 128, 128, 3), data_y.shape: (2520, 3)

# train/test가 나뉘어 있지 않으므로 나누어 주도록 함.

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=16)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

'''
ic| x_train.shape: (1890, 128, 128, 3)
    x_test.shape: (630, 128, 128, 3)
    y_train.shape: (1890, 3)
    y_test.shape: (630, 3)
'''

ic(y_train[:5], y_train[-5:])

model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(128, 128, 3), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
model.add(GlobalAvgPool2D())
model.add(Dense(256))
# model.add(Dropout(0.5))
model.add(Dense(64))
# model.add(Dropout(0.5))
model.add(Dense(16))
# model.add(Dropout(5/8))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# OneHot을 안 해줬다면 sparse_categorical_crossentopy로 해야 함. 

start_time = time.time()

model.fit(x_train, y_train, validation_split=1/8, validation_steps=25, epochs=10)

elapsed_time = time.time() - start_time

ic(elapsed_time)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
[Best Fit]
ic| elapsed_time: 11.587864875793457
16/16 [==============================] - 0s 14ms/step - loss: 1.0945 - accuracy: 0.3829
loss :  1.0944865942001343
accuracy :  0.3829365074634552
'''