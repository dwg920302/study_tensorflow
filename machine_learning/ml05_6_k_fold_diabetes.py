# Machine Learning #5 [K-Fold]

from icecream import ic
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score

from sklearn.datasets import load_diabetes

# KFold, Cross_Validation

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_diabetes()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

kfold = KFold(n_splits=8, shuffle=True, random_state=37)

# y 카테고리화 하지 않음 (auto)

# Model

model = LinearSVC()
# ic| scores: array([0.        , 0.02247191, 0.        , 0.        , 0.01136364])

# model = SVC()
# ic| scores: array([0.01123596, 0.01123596, 0.        , 0.        , 0.01136364])

# model = KNeighborsClassifier()
# ic| scores: array([0.        , 0.        , 0.        , 0.01136364, 0.        ])

# model = LogisticRegression()
# ic| scores: array([0.01123596, 0.01123596, 0.        , 0.        , 0.01136364])

# model = DecisionTreeClassifier()
# ic| scores: array([0.        , 0.02247191, 0.        , 0.01136364, 0.        ])

# model = RandomForestClassifier()
# ic| scores: array([0.01123596, 0.02247191, 0.        , 0.02272727, 0.        ])

scores = cross_val_score(model, x, y, cv=kfold)

ic(scores)

# ic(scores, round(np.mean(scores), 4))
