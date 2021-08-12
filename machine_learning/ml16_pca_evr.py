# 상관계수(相關係數, correlation coefficient)

from  icecream import ic
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


dataset = load_diabetes()
x = dataset.data
y = dataset.target
ic(x.shape, y.shape)    # ic| x.shape: (150, 4), y.shape: (150,)

pca = PCA(n_components=7)
x2 = pca.fit_transform(x)
ic(x2, x2.shape)

model = XGBRegressor()

model.fit(x2, y)

results = model.score(x2, y)
ic(results) # ic| results: 0.9999349120798557