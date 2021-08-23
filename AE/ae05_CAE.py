# Convolution Auto Encoder

import numpy as np
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D

import matplotlib.pyplot as plt

import random

from icecream import ic

(x_train, _), (x_test, _) = mnist.load_data()

ic(x_train.shape, x_test.shape)

x_tr_sh = x_train.shape

x_te_sh = x_test.shape

x_train = x_train.reshape(x_tr_sh[0], x_tr_sh[1], x_tr_sh[2], 1).astype('float')/255
x_test = x_test.reshape(x_te_sh[0], x_te_sh[1], x_te_sh[2], 1).astype('float')/255

def autoencoder_mp():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu'))
    return model

def autoencoder_us():
    model = Sequential()
    model.add(Conv2D(filters=1, kernel_size=(1, 1), input_shape=(28, 28, 1), activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(filters=4, kernel_size=(1, 1), activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(filters=16, kernel_size=(1, 1), activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(filters=1, activation='relu'))
    return model


# 이미지 5개 무작위 픽
random_images = random.sample(range(x_train.shape[0]), 5)

model = autoencoder_mp()

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output_1 = model.predict(x_test)


model = autoencoder_us()

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output_2 = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))



# 원본 이미지를 위에, 오토 인코더가 출력한 이미지는 아래에

for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i , ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output_1[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT_1", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i , ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output_2[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT_2", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()