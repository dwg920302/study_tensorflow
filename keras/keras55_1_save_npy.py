from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd


# dataset = load_iris()
# dataset = load_boston()
# dataset = load_breast_cancer()
# dataset = load_diabetes()

# dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
# print(dataset)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# x_data = dataset.data
# y_data = dataset.target

# x_data = dataset.drop(columns='quality')
# y_data = dataset['quality'].to_numpy()

# np.save('../_save/_npy/k55_x_data_wine.npy', arr=x_data)
# np.save('../_save/_npy/k55_y_data_wine.npy', arr=y_data)

np.save('../_save/_npy/k55_x_train_fashion_mnist.npy', arr=x_train)
np.save('../_save/_npy/k55_x_test_fashion_mnist.npy', arr=x_test)
np.save('../_save/_npy/k55_y_train_fashion_mnist.npy', arr=y_train)
np.save('../_save/_npy/k55_y_test_fashion_mnist.npy', arr=y_test)