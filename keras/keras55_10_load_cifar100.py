import numpy as np
from icecream import ic

x_train = np.load('../_save/_npy/k55_x_train_cifar100.npy')
y_train = np.load('../_save/_npy/k55_y_train_cifar100.npy')
x_test = np.load('../_save/_npy/k55_x_test_cifar100.npy')
y_test = np.load('../_save/_npy/k55_y_test_cifar100.npy')

ic(x_train.shape, y_train.shape)
ic(x_test.shape, y_test.shape)

# ic| x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
# ic| x_test.shape: (10000, 28, 28), y_test.shape: (10000,)