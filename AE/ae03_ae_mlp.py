# 2개의 모델을 구성, 하나는 기본적 오토 인코더, 다른 하나는 더 딥하게 만든 것

import numpy as np
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

import matplotlib.pyplot as plt

import random

from icecream import ic

(x_train, _), (x_test, _) = mnist.load_data()

ic(x_train.shape, x_test.shape)

x_tr_sh = x_train.shape

x_te_sh = x_test.shape

x_train = x_train.reshape(x_tr_sh[0], x_tr_sh[1] * x_tr_sh[2]).astype('float')/255
x_test = x_test.reshape(x_te_sh[0], x_te_sh[1] * x_te_sh[2]).astype('float')/255

def autoencoder_shallow(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder_deep(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder_shallow(hidden_layer_size=154)  # pca 95%

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개 무작위 픽
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 위에, 오토 인코더가 출력한 이미지는 아래에

for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i , ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


model = autoencoder_deep(hidden_layer_size=154)  # pca 95%

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개 무작위 픽

# 원본 이미지를 위에, 오토 인코더가 출력한 이미지는 아래에

for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i , ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()