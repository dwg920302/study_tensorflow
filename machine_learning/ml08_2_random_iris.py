from icecream import ic
import warnings
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},  # 4 * 1 * 5 = 20
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},  # 3 * 1 * 2 * 5 = 30
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} #   4 * 1 * 2 * 5 = 40
]   # 총 20+30+40 = 90번 연산. 5는 kfold의 split
# degree?

# RamdomizedSearchCV로 감싸기

model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1)

model.fit(x_train, y_train)

ic(model.best_estimator_)   # ic| model.best_estimator_: SVC(C=1, kernel='linear')

ic(model.best_score_)   # ic| model.best_score_: 0.9800000000000001

ic(model.score(x_test, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
ic| model.best_estimator_: SVC(C=1, kernel='linear')
ic| model.best_score_: 0.975
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
'''