# Machine Learning #5 [K-Fold]

from icecream import ic
import warnings
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

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

# Model

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},  # 4 * 1 * 5 = 20
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},  # 3 * 1 * 2 * 5 = 30
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} #   4 * 1 * 2 * 5 = 40
]   # 총 20+30+40 = 90번 연산. 5는 kfold의 split

# GridSearchCV로 감싸기

model = GridSearchCV(SVC(), parameters, cv=kfold)
# ic| scores: array([1.        , 1.        , 0.96666667, 0.83333333, 1.        ])

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
