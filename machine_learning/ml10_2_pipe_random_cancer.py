from icecream import ic
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, accuracy_score

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline


dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

# parameters = [
#     {"C":[1, 10, 100, 1000], "kernel":["linear"]},  # 4 * 1 * 5 = 20
#     {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},  # 3 * 1 * 2 * 5 = 30
#     {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} #   4 * 1 * 2 * 5 = 40
# ]   # 총 20+30+40 = 90번 연산. 5는 kfold의 split
# # degree?

parameters =[
    {'randomforestclassifier__min_samples_leaf' : [3, 5, 7],
    'randomforestclassifier__max_depth' : [2, 3, 5, 10],
    'randomforestclassifier__min_samples_split' : [6, 8, 10]}
]

# RamdomizedSearchCV로 감싸기

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

model.fit(x_train, y_train)

ic(model.best_estimator_)   # ic| model.best_estimator_: SVC(C=1, kernel='linear')

ic(model.best_params_)

ic(model.best_score_)   # ic| model.best_score_: 0.9800000000000001

ic(model.score(x_test, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

'''
Fitting 5 folds for each of 36 candidates, totalling 180 fits
ic| model.best_estimator_: Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                                           ('randomforestclassifier',
                                            RandomForestClassifier(max_depth=5, min_samples_leaf=3,
                                                                   min_samples_split=6))])
ic| model.best_params_: {'randomforestclassifier__max_depth': 5,
                         'randomforestclassifier__min_samples_leaf': 3,
                         'randomforestclassifier__min_samples_split': 6}
ic| model.best_score_: 0.964835164835165
ic| model.score(x_test, y_test): 0.9473684210526315
정답률 :  0.9473684210526315
'''