from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes
import numpy as np
import pandas as pd


# dataset = load_iris()
# dataset = load_boston()
# dataset = load_breast_cancer()
# dataset = load_diabetes()

dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)


# x_data = dataset.data
# y_data = dataset.target

x_data = dataset.drop(columns='quality')
y_data = dataset['quality'].to_numpy()

np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data)