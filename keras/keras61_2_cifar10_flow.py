# 훈련 데이터를 10만개로 증폭하고, 완료 후 기존 모델과 비교
# save_dir도 temp에 넣어볼 것

from tensorflow.keras.datasets import cifar10
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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ic(x_train.shape)

augment_size=10

randidx=np.random.randint(x_train.shape[0], size=augment_size)
ic(x_train.shape[0])
ic(randidx, randidx.shape)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(x_augmented,
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False,
                                save_to_dir='../temp/').next()[0]

# 10개의 샘플 뽑아보기

augment_size=40000

randidx=np.random.randint(x_train.shape[0], size=augment_size)
ic(randidx, randidx.shape)
x_augmented = x_train[randidx].copy()
x_augmented = train_datagen.flow(x_augmented,
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]

y_augmented = y_train[randidx].copy()

# train에 병합할 데이터 40000개

ic(x_augmented.shape)
# ic| x_augmented.shape: (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))    # 60000 + 40000

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(32, 32, 3), padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
model.add(GlobalAvgPool2D())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(5/8))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=1/0, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
ic(loss, accuracy)

# ic| loss: nan, accuracy: 0.10000000149011612