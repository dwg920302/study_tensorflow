# Machine Learning #5 [K-Fold]

from icecream import ic
import warnings
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, KFold, cross_val_score

# KFold, Cross_Validation

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_iris()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

# y 카테고리화 하지 않음 (auto)

# Model

# model = LinearSVC()
# ic| scores: array([1.        , 1.        , 0.83333333, 0.83333333, 1.        ])

# model = SVC()
# ic| scores: array([1.        , 1.        , 1.        , 0.83333333, 1.        ])

# model = KNeighborsClassifier()
# ic| scores: array([1.        , 1.        , 1.        , 0.83333333, 1.        ])

# model = LogisticRegression()
# ic| scores: array([1.        , 1.        , 1.        , 0.83333333, 1.        ])

# model = DecisionTreeClassifier()
# ic| scores: array([1.        , 1.        , 0.83333333, 0.83333333, 1.        ])

model = RandomForestClassifier()
# ic| scores: array([0.83333333, 1.        , 1.        , 1.        , 1.        ])

scores = cross_val_score(model, x_train, y_train, cv=kfold)

ic(scores)

# ic(scores, round(np.mean(scores), 4))
