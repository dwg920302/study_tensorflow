from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

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

augment_size=100

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),   # x값 똑같은 것들
    np.zeros(augment_size),    # y값 똑같은 것들
    batch_size=augment_size,
    shuffle=False
)

ic(type(x_data))
ic(type(x_data[0]))
ic(type(x_data[0][0]))
ic(x_data[0][0].shape)
ic(x_data[0][1].shape)
'''
ic| type(x_data): <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
ic| type(x_data[0]): <class 'tuple'>
ic| type(x_data[0][0]): <class 'numpy.ndarray'>
ic| x_data[0][0].shape: (100, 28, 28, 1)
ic| x_data[0][1].shape: (100,)
'''

plt.figure(figsize=(7,7))
for i in range(1, 49+1):
    plt.subplot(7,7,i)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),   # x값 똑같은 것들
    np.zeros(augment_size),    # y값 똑같은 것들
    batch_size=augment_size,
    shuffle=False
).next()

ic(type(x_data))
ic(type(x_data[0]), type(x_data[1]))
ic(x_data[0].shape)
ic(x_data[1].shape)
'''
ic| type(x_data): <class 'tuple'>
ic| type(x_data[0]): <class 'numpy.ndarray'>
    type(x_data[1]): <class 'numpy.ndarray'>
ic| x_data[0].shape: (100, 28, 28, 1)
ic| x_data[1].shape: (100,)
'''

plt.figure(figsize=(7,7))
for i in range(1, 49+1):
    plt.subplot(7,7,i)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()
