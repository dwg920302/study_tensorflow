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

pca_evr = pca.explained_variance_ratio_
ic(pca_evr, sum(pca_evr)) # feature_importance와 비슷하나 다르며, 합이 n_components에 따라 달라짐 (무조건 1이 아님)
# 기본적으로 n_components가 커질수록 1에 가까워지며, n_components == 원래 column 수일 경우에는 1
cumsum = np.cumsum(pca_evr)
ic(cumsum) # 누적합

# ic| pca_evr: array([0.40242142, 0.14923182, 0.12059623, 0.09554764, 0.06621856,
#                     0.06027192, 0.05365605])
#     sum(pca_evr): 0.9479436357350414
# ic| cumsum: array([0.40242142, 0.55165324, 0.672

pca = PCA(n_components=10)
x2 = pca.fit_transform(x)
ic(x2, x2.shape)

pca_evr = pca.explained_variance_ratio_
ic(pca_evr, sum(pca_evr))

cumsum = np.cumsum(pca_evr)
ic(cumsum) 
# ic| pca_evr: array([0.40242142, 0.14923182, 0.12059623, 0.09554764, 0.06621856,
#                     0.06027192, 0.05365605, 0.04336832, 0.00783199, 0.00085605])
#     sum(pca_evr): 1.0
# ic| cumsum: array([0.40242142, 0.55165324, 0.67224947, 0.76779711, 0.83401567,
#                    0.89428759, 0.94794364, 0.99131196, 0.99914395, 1.        ])

ic(np.argmax(cumsum >= 0.94)+1)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()