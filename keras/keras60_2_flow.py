from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, GlobalAvgPool2D

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest'
)

# ImageDataGenerator를 정의

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 파일에서 땡겨오려면 flow_from_directory() // x, y가  튜플 형태로 뭉쳐 있음
# 데이터에서 땡겨오려면 flow()  // x, y가 나뉘어 있음

augment_size=40000

randidx=np.random.randint(x_train.shape[0], size=augment_size)
ic(x_train.shape[0])
# ic| x_train.shape[0]: 60000
ic(randidx, randidx.shape)
# ic| randidx: array([58843, 15235, 13618, ...,  9159,  3154, 17436])
# randidx.shape: (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(x_augmented.reshape(augment_size, 28, 28, 1), np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

ic(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)

model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(28, 28, 1), padding='same'))
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])