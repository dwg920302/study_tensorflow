from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
import matplotlib.pyplot as plt

datasets = load_diabetes()
x = datasets.data
y = datasets.target

# 데이터 전처리(preprocess)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()

# scaler.fit(x)
# x = scaler.transform(x)

x_plot = np.transpose(x)

for x in x_plot:
    plt.scatter(x, y)
    plt.show()