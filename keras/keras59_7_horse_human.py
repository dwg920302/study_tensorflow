from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAvgPool2D, MaxPool2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from icecream import ic
import time
from tensorflow.python.keras.layers.core import Dropout

# label 2개지만, 다중분류로도 풀어보기

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
    '../_data/horse_or_human',
    target_size=(256, 256),
    batch_size=5000,
    class_mode='categorical',
    shuffle=True
)
# Found 3309 images belonging to 2 classes.

ic(data)

start_time = time.time()

# data_x = data[0][0] # 이미지파일 같음, 데이터 할당하는데 은근 오래 걸림 (2~3분 정도)
# data_y = data[0][1] # 인덱스 같음 (남=0 여=1)

# np.save('../_save/_npy/k59_x_data_horsehuman.npy', arr=data_x)
# np.save('../_save/_npy/k59_y_data_horsehuman.npy', arr=data_y)

data_x = np.load('../_save/_npy/k59_x_data_horsehuman.npy')
data_y = np.load('../_save/_npy/k59_y_data_horsehuman.npy')

elapsed_time_get_data = time.time() - start_time

ic(elapsed_time_get_data)   
ic(data_x.shape, data_y.shape)

# ic| data_x.shape: (3309, 256, 256, 3), data_y.shape: (3309,)

# train/test가 나뉘어 있지 않으므로 나누어 주도록 함.

# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, shuffle=True, random_state=24)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=49)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

'''
ic| x_train.shape: (2481, 256, 256, 3)
    x_test.shape: (828, 256, 256, 3)
    y_train.shape: (2481,)
    y_test.shape: (828,)
'''

ic(y_train[:5], y_train[-5:])

model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(256, 256, 3), padding='same'))   # 256 하니까 계속 터짐 ㅡㅡ
model.add(MaxPool2D())
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
model.add(GlobalAvgPool2D())
model.add(Dense(256))
# model.add(Dropout(0.5))
model.add(Dense(64))
# model.add(Dropout(0.5))
model.add(Dense(16))
# model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# hist = model.fit_generator(xy_train, validation_data=xy_test, validation_steps=5, epochs=10)

start_time = time.time()

model.fit(x_train, y_train, validation_split=1/8, validation_steps=25, epochs=10)

elapsed_time = time.time() - start_time

ic(elapsed_time)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
ic| elapsed_time: 17.26269245147705
7/7 [==============================] - 0s 26ms/step - loss: 0.6052 - accuracy: 0.6748
loss :  0.6051723957061768
accuracy :  0.6747573018074036
'''