from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

print(x_train[7], ' : ', y_train[7])

plt.imshow(x_train[7], 'viridis')
plt.show()

print(np.unique(y_train))