import numpy as np
from icecream import ic

x_train = np.load('../_save/_npy/k59_brain_x_train.npy')
y_train = np.load('../_save/_npy/k59_brain_y_train.npy')
x_test = np.load('../_save/_npy/k59_brain_x_test.npy')
y_test = np.load('../_save/_npy/k59_brain_y_test.npy')

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)