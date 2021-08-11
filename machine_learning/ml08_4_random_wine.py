from icecream import ic
import warnings
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# KFold, Cross_Validation

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

start_time = time()

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)

model.fit(x_train, y_train)

ic(model.best_estimator_)

ic(model.best_score_)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

elapsed_time = time() - start_time

ic(elapsed_time)

'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
ic| model.best_estimator_: RandomForestClassifier(n_jobs=2)
ic| model.best_score_: 0.6705059035108295
ic| model.score(x_test, y_test): 0.7030612244897959
정답률 :  0.7030612244897959
ic| elapsed_time: 22.24638342857361
'''