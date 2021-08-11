from icecream import ic
import warnings
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# KFold, Cross_Validation

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=41)

# Model

kfold = KFold(n_splits=5, shuffle=True, random_state=41)

parameters = [
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

'''
parameters = [
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]
'''

# GridSearchCV로 감싸기

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

model.fit(x_train, y_train)

ic(model.best_estimator_)

ic(model.best_score_)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

'''
ic| model.best_estimator_: RandomForestClassifier(min_samples_split=3)
ic| model.best_score_: 0.9582417582417582
ic| model.score(x_test, y_test): 0.9912280701754386
정답률 :  0.9912280701754386
'''