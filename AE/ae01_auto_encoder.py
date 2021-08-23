import numpy as np
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import matplotlib.pyplot as plt

from icecream import ic


(x_train, _), (x_test, _) = mnist.load_data()

ic(x_train.shape, x_test.shape)

x_tr_sh = x_train.shape

x_te_sh = x_test.shape

x_train = x_train.reshape(x_tr_sh[0], x_tr_sh[1] * x_tr_sh[2]).astype('float')/255
x_test = x_test.reshape(x_te_sh[0], x_te_sh[1] * x_te_sh[2]).astype('float')/255

input_img = Input(shape=x_tr_sh[1] * x_tr_sh[2], )
encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)
decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

decoded_img = autoencoder.predict(x_test)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()