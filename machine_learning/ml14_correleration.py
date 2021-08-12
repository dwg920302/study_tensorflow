# 상관계수(相關係數, correlation coefficient)

from  icecream import ic
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


dataset = load_iris()
ic(dataset.keys())  # ic| dataset.keys(): dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
ic(dataset.target_names)    # ic| dataset.target_names: array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

x = dataset.data
y = dataset.target
ic(x.shape, y.shape)    # ic| x.shape: (150, 4), y.shape: (150,)


df = pd.DataFrame(x, columns=dataset['feature_names'])
ic(df)

# y컬럼 추가
df['Target'] = y
ic(df.head())

print("[상관계수 히트 맵]")
ic(df.corr())
'''
ic| df.corr():                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
               sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
               sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
               petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
               petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
               Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
'''

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()