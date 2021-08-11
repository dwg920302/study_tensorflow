# Machine Learning #5 [K-Fold]

from icecream import ic
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score

# KFold, Cross_Validation

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality'].to_numpy()
x = dataset.drop(columns='quality')

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

# y 카테고리화 하지 않음 (auto)

# Model

model = LinearSVC()
# ic| scores: array([0.35816327, 0.33163265, 0.29183673, 0.46475996, 0.47191011])

# model = SVC()
# ic| scores: array([0.96666667, 1.        , 1.        , 0.9       , 1.        ])

# model = KNeighborsClassifier()
# ic| scores: array([0.96666667, 1.        , 0.96666667, 0.9       , 1.        ])

# model = LogisticRegression()
# ic| scores: array([1.        , 1.        , 0.96666667, 0.86666667, 1.        ])

# model = DecisionTreeClassifier()
# ic| scores: array([0.96666667, 0.96666667, 0.96666667, 0.9       , 1.        ])

# model = RandomForestClassifier()
# ic| scores: array([0.96666667, 1.        , 0.96666667, 0.9       , 1.        ])

scores = cross_val_score(model, x, y, cv=kfold)

ic(scores)

# ic(scores, round(np.mean(scores), 4))
