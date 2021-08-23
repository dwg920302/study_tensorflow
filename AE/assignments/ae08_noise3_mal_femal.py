# male/female 데이터에 Noise를 넣어서, 기미, 주근깨, 여드름 (대충 점 같은 거)을 제거하기
# HW

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from icecream import ic
from time import time


test_datagen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.5,
    shear_range=0.5,
    fill_mode='nearest'
)

# train 내부 폴더 -> y의 라벨로 자동 지정됨
# x, y 자동 생성
data = datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(128, 128),
    batch_size=5000,
    class_mode='binary',
    shuffle=True
)
# Found 3309 images belonging to 2 classes.

ic(data)

start_time = time()

data_x = data[0][0] # 이미지파일 같음, 데이터 할당하는데 은근 오래 걸림 (2~3분 정도)
data_y = data[0][1] # 인덱스 같음 (남=0 여=1)

np.save('../_save/_npy/k59_x_data_menwomen_128.npy', arr=data_x)
np.save('../_save/_npy/k59_y_data_menwomen_128.npy', arr=data_y)

# 256 256으로 돌리니 터져서 128으로 npy도 다시 작성하고 함

data_x = np.load('../_save/_npy/k59_x_data_menwomen_128.npy')
data_y = np.load('../_save/_npy/k59_y_data_menwomen_128.npy')

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=87)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

augment_size=int(x_test.shape[0]/5)

randidx=np.random.randint(augment_size, size=augment_size)
ic(randidx, randidx.shape)

x_augmented = x_test[randidx].copy()
y_augmented = y_test[randidx].copy()

x_augmented = test_datagen.flow(x_augmented,
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False,
                                save_to_dir='../temp/').next()[0]

x_test = np.concatenate((x_test, x_augmented))
y_test = np.concatenate((y_test, y_augmented))

x_train_noised = x_train + np.random.normal(0, 0.2, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.2, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

ic(x_test.shape, y_test.shape)

'''
ic| x_train.shape: (2647, 128, 128, 3)
    x_test.shape: (662, 128, 128, 3)
    y_train.shape: (2647,)
    y_test.shape: (662,)
'''

def auto_encoder():
    model = Sequential()
    model.add(Input(shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(1, 1), padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(1, 1)))
    model.add(Conv2D(3, (1, 1), activation='sigmoid', padding='same'))
    return model

model = auto_encoder()


model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=25, batch_size=256)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))


random_images = random.sample(range(output.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(128, 128, 3))
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(128, 128, 3))
    if i == 0:
        ax.set_ylabel('noise', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(128, 128, 3), cmap='rainbow')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])



plt.show()