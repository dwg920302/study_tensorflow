# Machine Learning #6 [KFold + all_estimators]

# 그냥 4+5한 것

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     # 분류 모델 (회귀 아님. 절대.)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# from sklearn.utils.testing import all_estimators  # (0.24 이전 버전에서)
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=61)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y 카테고리화 하지 않음 (auto)

# Model

allAlgorithms_cl = all_estimators(type_filter='classifier')   # 모든 모델
# ic(allAlgorithms_cl)
# allAlgorithms_rg = all_estimators(type_filter='regressor')
# ic(allAlgorithms_rg)

kfold = KFold(n_splits=5, shuffle=True, random_state=61)

for (name, algorithm) in allAlgorithms_cl:
    try:
        model = algorithm()
        
        scores = cross_val_score(model, x, y, cv=kfold)

        ic(name, scores)

        # model.fit(x_train, y_train)

        # y_predict = model.predict(x_test)
        # acc_score = accuracy_score(y_test, y_predict)
        # ic(name, acc_score)
    except Exception as e:
        # ic(e)
        print(name, '은 오류가 나서 실행하지 않음')
        continue

ic(len(allAlgorithms_cl))    # ic| len(allAlgorithms_cl): 41, len(allAlgorithms_rg): 54

'''
ic| name: 'AdaBoostClassifier'
    scores: array([0.96491228, 0.96491228, 0.96491228, 0.98245614, 0.92920354])
ic| name: 'BaggingClassifier'
    scores: array([0.95614035, 0.92982456, 0.96491228, 0.97368421, 0.9380531 ])
ic| name: 'BernoulliNB'
    scores: array([0.59649123, 0.64912281, 0.65789474, 0.62280702, 0.61061947])
ic| name: 'CalibratedClassifierCV'
    scores: array([0.92982456, 0.9122807 , 0.93859649, 0.93859649, 0.91150442])
ic| name: 'CategoricalNB'
    scores: array([       nan,        nan, 0.95614035,        nan,        nan])
ClassifierChain 은 오류가 나서 실행하지 않음
ic| name: 'ComplementNB'
    scores: array([0.89473684, 0.88596491, 0.92982456, 0.92105263, 0.84955752])
ic| name: 'DecisionTreeClassifier'
    scores: array([0.9122807 , 0.88596491, 0.94736842, 0.94736842, 0.90265487])
ic| name: 'DummyClassifier'
    scores: array([0.59649123, 0.64912281, 0.65789474, 0.62280702, 0.61061947])
ic| name: 'ExtraTreeClassifier'
    scores: array([0.96491228, 0.89473684, 0.94736842, 0.92982456, 0.89380531])
ic| name: 'ExtraTreesClassifier'
    scores: array([0.97368421, 0.92982456, 0.99122807, 0.95614035, 0.96460177])
ic| name: 'GaussianNB'
    scores: array([0.92982456, 0.9122807 , 0.98245614, 0.95614035, 0.90265487])
ic| name: 'GaussianProcessClassifier'
    scores: array([0.94736842, 0.83333333, 0.96491228, 0.9122807 , 0.91150442])
ic| name: 'GradientBoostingClassifier'
    scores: array([0.98245614, 0.96491228, 0.97368421, 0.97368421, 0.94690265])
ic| name: 'HistGradientBoostingClassifier'
    scores: array([0.97368421, 0.97368421, 0.95614035, 0.97368421, 0.96460177])
ic| name: 'KNeighborsClassifier'
    scores: array([0.92982456, 0.87719298, 0.97368421, 0.93859649, 0.92920354])
ic| name: 'LabelPropagation'
    scores: array([0.43859649, 0.36842105, 0.38596491, 0.40350877, 0.40707965])
ic| name: 'LabelSpreading'
    scores: array([0.43859649, 0.36842105, 0.38596491, 0.40350877, 0.40707965])
ic| name: 'LinearDiscriminantAnalysis'
    scores: array([0.96491228, 0.94736842, 0.99122807, 0.95614035, 0.9380531 ])
ic| name: 'LinearSVC'
    scores: array([0.9122807 , 0.9122807 , 0.95614035, 0.83333333, 0.90265487])
ic| name: 'LogisticRegression'
    scores: array([0.93859649, 0.93859649, 0.99122807, 0.96491228, 0.9380531 ])
ic| name: 'LogisticRegressionCV'
    scores: array([0.93859649, 0.97368421, 0.99122807, 0.95614035, 0.94690265])
ic| name: 'MLPClassifier'
    scores: array([0.92982456, 0.90350877, 0.97368421, 0.92982456, 0.90265487])
MultiOutputClassifier 은 오류가 나서 실행하지 않음
ic| name: 'MultinomialNB'
    scores: array([0.89473684, 0.89473684, 0.92982456, 0.92105263, 0.84955752])
ic| name: 'NearestCentroid'
    scores: array([0.89473684, 0.88596491, 0.92982456, 0.90350877, 0.84070796])
ic| name: 'NuSVC'
    scores: array([0.85087719, 0.85964912, 0.92982456, 0.89473684, 0.83185841])
OneVsOneClassifier 은 오류가 나서 실행하지 않음
OneVsRestClassifier 은 오류가 나서 실행하지 않음
OutputCodeClassifier 은 오류가 나서 실행하지 않음
ic| name: 'PassiveAggressiveClassifier'
    scores: array([0.92105263, 0.74561404, 0.93859649, 0.92982456, 0.91150442])
ic| name: 'Perceptron'
    scores: array([0.85087719, 0.81578947, 0.95614035, 0.89473684, 0.45132743])
ic| name: 'QuadraticDiscriminantAnalysis'
    scores: array([0.96491228, 0.92105263, 0.97368421, 0.94736842, 0.96460177])
ic| name: 'RadiusNeighborsClassifier'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'RandomForestClassifier'
    scores: array([0.95614035, 0.92105263, 0.98245614, 0.97368421, 0.95575221])
ic| name: 'RidgeClassifier'
    scores: array([0.96491228, 0.96491228, 0.96491228, 0.93859649, 0.92920354])
ic| name: 'RidgeClassifierCV'
    scores: array([0.96491228, 0.96491228, 0.96491228, 0.95614035, 0.9380531 ])
ic| name: 'SGDClassifier'
    scores: array([0.75438596, 0.79824561, 0.49122807, 0.92982456, 0.90265487])
ic| name: 'SVC'
    scores: array([0.9122807 , 0.88596491, 0.94736842, 0.92982456, 0.89380531])
StackingClassifier 은 오류가 나서 실행하지 않음
VotingClassifier 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_cl): 41
'''

# model.save('../_save/ml04_1_iris.h5')