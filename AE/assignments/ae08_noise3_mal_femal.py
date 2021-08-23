# male/female 데이터에 Noise를 넣어서, 기미, 주근깨, 여드름 (대충 점 같은 거)을 제거하기
# HW

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Conv2DTranspose, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from icecream import ic


test_datagen = ImageDataGenerator(rescale=1./255)

data_x = np.load('../_save/_npy/k59_x_data_menwomen.npy')
data_y = np.load('../_save/_npy/k59_y_data_menwomen.npy')

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=91)

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

ic(x_test.shape, y_test.shape)

'''
ic| x_train.shape: (2647, 256, 256, 3)
    x_test.shape: (662, 256, 256, 3)
    y_train.shape: (2647,)
    y_test.shape: (662,)
'''

def autoEncoder():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x_1 = x

    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x_2 = x

    x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2)
    x = Dropout(0.4)(x)
    # x = LeakyReLU()(x)
    x = x

    x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same', activation='sigmoid')(x+x_1)
    x = Dropout(0.4)(x)
    x = x
    outputs = x
    model = Model(inputs = inputs,outputs=outputs)

    return model

model1 = autoEncoder()

# model1.summary()

# 3. compile, train
model1.compile(optimizer='adam', loss='mse')

model1.fit(x_train, x_train, epochs=10, batch_size=512)

# 4. eval pred
output1 = model1.predict(x_test)

# 5. visualize
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10) ) = \
    plt.subplots(2, 5, figsize = (20, 7))


# 이미지 다섯 개 무작위
random_images = random.sample(range(output1.shape[0]), 5)

# 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# basic 오토인코더가 출력한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('basic', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()