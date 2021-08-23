import numpy as np
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
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

# x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
# x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = x_train + np.random.normal(0, 0.3, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.3, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)    # 값들을 0에서 1 사이로 변경

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)  # pca 95%

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax21, ax22, ax23, ax24, ax25)) = plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개 무작위 픽
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 위에, 오토 인코더가 출력한 이미지는 아래에

for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='brg')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i , ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='brg')
    if i == 0:
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i , ax in enumerate([ax21, ax22, ax23, ax24, ax25]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='brg')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()