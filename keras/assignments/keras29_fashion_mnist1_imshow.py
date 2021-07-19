from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

print(x_train[12], ' : ', y_train[12]) 

plt.imshow(x_train[12], 'viridis')
plt.show()

print(np.unique(y_train))

