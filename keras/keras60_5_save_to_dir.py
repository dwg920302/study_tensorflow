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

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

ic(x_train.shape)

augment_size=10

randidx=np.random.randint(x_train.shape[0], size=augment_size)
ic(x_train.shape[0])
# ic| x_train.shape[0]: 60000
ic(randidx, randidx.shape)
# ic| randidx: array([58843, 15235, 13618, ...,  9159,  3154, 17436])
# randidx.shape: (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)

# x_augmented = train_datagen.flow(x_augmented,
#                                 np.zeros(augment_size),
#                                 batch_size=augment_size,
#                                 shuffle=False,
#                                 save_to_dir='../temp/').next()[0]

# save_to_dir -> 해당 경로에 뽑아온 파일을 저장. batch_size가 augment_size = 10 이므로 10장을 뽑아옴, 수치 변경 가능
# 이 구문이 실행되는 대로 바로 출력

x_augmented = train_datagen.flow(x_augmented,
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False,
                                save_to_dir='../temp/')

# 분명 10개인데, 왜 몇십개가 나올까?
# 이 구문을 실행한 시점에서는 생기지 않고, 아래에서 x_augmented 를 추가로 호출하는 만큼 이미지가 계속 생성됨

ic(type(x_augmented), x_augmented) # ic| type(x_augmented): <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
ic(type(x_augmented[0]), x_augmented[0]) # 이 시점에서 2번(10개)

print(x_augmented[0][0].shape)  # 이 시점에서 1번(10개)
print(x_augmented[0][1].shape)  # 이 시점에서 1번(10개)
# print(x_augmented[0][1][:4])