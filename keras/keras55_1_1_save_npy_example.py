from sklearn.datasets import load_iris
import numpy as np


dataset = load_iris()

x_data = dataset.data
y_data = dataset.target

np.save('./_save/_npy/k55_x_data.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data.npy', arr=y_data)