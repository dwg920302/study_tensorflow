from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt


# 실습 1
# x_augmented 10개와 원래 x_train 10개를 비교하는 이미지를 출력할 것.
# subplot(2, 10, ?) 사용

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

ic(x_train.shape)

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


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

ic(x_train.shape, x_augmented.shape)
# ic| x_augmented.shape: (40000, 28, 28, 1)

print(x_train.shape, y_train.shape)

plt.figure(figsize=(2, 10))
for i in range(1, 10+1):
    plt.subplot(2, 10, i)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')
    plt.subplot(2, 10, 10+i)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()

# 같이 했을 때 2번째 그림만 나옴, 덮어씌인거 같음.