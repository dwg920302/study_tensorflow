# Machine Learning #6 [KFold + all_estimators]

# 그냥 4+5한 것

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_iris
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


dataset = load_iris()

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

'''
tuple ('classifier' 이름, 'classifier' 클래스)

ic| allAlgorithms_cl: [('AdaBoostClassifier',
                      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),
                     ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>),
                     ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>),
                     ('CalibratedClassifierCV',
                      <class 'sklearn.calibration.CalibratedClassifierCV'>),
                     ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>),
                     ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>),
                     ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>),
                     ('DecisionTreeClassifier',
                      <class 'sklearn.tree._classes.DecisionTreeClassifier'>),
                     ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>),
                     ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>),
                     ('ExtraTreesClassifier',
                      <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>),
                     ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>),
                     ('GaussianProcessClassifier',
                      <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>),
                     ('GradientBoostingClassifier',
                      <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>),
                     ('HistGradientBoostingClassifier',
                      <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>),
                     ('KNeighborsClassifier',
                      <class 'sklearn.neighbors._classification.KNeighborsClassifier'>),
                     ('LabelPropagation',
                      <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>),
                     ('LabelSpreading',
                      <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>),
                     ('LinearDiscriminantAnalysis',
                      <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>),
                     ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>),
                     ('LogisticRegression',
                      <class 'sklearn.linear_model._logistic.LogisticRegression'>),
                     ('LogisticRegressionCV',
                      <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>),
                     ('MLPClassifier',
                      <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>),
                     ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>),
                     ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>),
                     ('NearestCentroid',
                      <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>),
                     ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>),
                     ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>),
                     ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>),
                     ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>),
                     ('PassiveAggressiveClassifier',
                      <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>),
                     ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>),
                     ('QuadraticDiscriminantAnalysis',
                      <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>),
                     ('RadiusNeighborsClassifier',
                      <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>),
                     ('RandomForestClassifier',
                      <class 'sklearn.ensemble._forest.RandomForestClassifier'>),
                     ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>),
                     ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>),
                     ('SGDClassifier',
                      <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>),
                     ('SVC', <class 'sklearn.svm._classes.SVC'>),
                     ('StackingClassifier',
                      <class 'sklearn.ensemble._stacking.StackingClassifier'>),
                     ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)
'''

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
    scores: array([0.93333333, 0.9       , 0.96666667, 0.96666667, 0.93333333])
ic| name: 'BaggingClassifier'
    scores: array([0.9       , 0.9       , 1.        , 1.        , 0.93333333])
ic| name: 'BernoulliNB'
    scores: array([0.3       , 0.33333333, 0.3       , 0.3       , 0.26666667])
ic| name: 'CalibratedClassifierCV'
    scores: array([0.93333333, 0.86666667, 1.        , 0.86666667, 0.9       ])
ic| name: 'CategoricalNB'
    scores: array([0.9       , 0.9       , 0.93333333, 0.93333333, 0.96666667])
ClassifierChain 은 오류가 나서 실행하지 않음
ic| name: 'ComplementNB'
    scores: array([0.63333333, 0.66666667, 0.63333333, 0.66666667, 0.73333333])
ic| name: 'DecisionTreeClassifier'
    scores: array([0.93333333, 0.9       , 0.96666667, 0.96666667, 0.93333333])
ic| name: 'DummyClassifier'
    scores: array([0.3       , 0.33333333, 0.3       , 0.3       , 0.26666667])
ic| name: 'ExtraTreeClassifier'
    scores: array([0.93333333, 0.9       , 1.        , 0.93333333, 1.        ])
ic| name: 'ExtraTreesClassifier'
    scores: array([0.96666667, 0.9       , 0.96666667, 1.        , 0.93333333])
ic| name: 'GaussianNB'
    scores: array([0.93333333, 0.9       , 1.        , 0.96666667, 0.93333333])
ic| name: 'GaussianProcessClassifier'
    scores: array([0.93333333, 0.9       , 1.        , 1.        , 0.93333333])
ic| name: 'GradientBoostingClassifier'
    scores: array([0.96666667, 0.9       , 0.96666667, 0.96666667, 0.93333333])
ic| name: 'HistGradientBoostingClassifier'
    scores: array([0.96666667, 0.9       , 0.96666667, 0.96666667, 0.9       ])
ic| name: 'KNeighborsClassifier'
    scores: array([0.96666667, 0.93333333, 0.96666667, 1.        , 0.93333333])
ic| name: 'LabelPropagation'
    scores: array([0.96666667, 0.93333333, 0.96666667, 1.        , 0.93333333])
ic| name: 'LabelSpreading'
    scores: array([0.96666667, 0.93333333, 0.96666667, 1.        , 0.93333333])
ic| name: 'LinearDiscriminantAnalysis'
    scores: array([1.        , 0.93333333, 1.        , 1.        , 0.93333333])
ic| name: 'LinearSVC'
    scores: array([1.        , 0.93333333, 1.        , 0.93333333, 0.93333333])
ic| name: 'LogisticRegression'
    scores: array([0.96666667, 0.9       , 1.        , 1.        , 0.93333333])
ic| name: 'LogisticRegressionCV'
    scores: array([0.96666667, 0.9       , 1.        , 1.        , 0.96666667])
ic| name: 'MLPClassifier'
    scores: array([0.96666667, 0.93333333, 0.96666667, 0.96666667, 0.96666667])
MultiOutputClassifier 은 오류가 나서 실행하지 않음
ic| name: 'MultinomialNB'
    scores: array([0.9       , 0.9       , 0.96666667, 0.96666667, 0.86666667])
ic| name: 'NearestCentroid'
    scores: array([0.9       , 0.93333333, 0.96666667, 0.9       , 0.96666667])
ic| name: 'NuSVC'
    scores: array([0.93333333, 0.9       , 0.96666667, 1.        , 0.93333333])
OneVsOneClassifier 은 오류가 나서 실행하지 않음
OneVsRestClassifier 은 오류가 나서 실행하지 않음
OutputCodeClassifier 은 오류가 나서 실행하지 않음
ic| name: 'PassiveAggressiveClassifier'
    scores: array([0.96666667, 0.66666667, 1.        , 0.86666667, 0.86666667])
ic| name: 'Perceptron'
    scores: array([0.96666667, 0.86666667, 0.83333333, 0.56666667, 0.7       ])
ic| name: 'QuadraticDiscriminantAnalysis'
    scores: array([0.96666667, 0.93333333, 1.        , 1.        , 0.96666667])
ic| name: 'RadiusNeighborsClassifier'
    scores: array([0.9       , 0.93333333, 1.        , 0.96666667, 0.9       ])
ic| name: 'RandomForestClassifier'
    scores: array([0.96666667, 0.9       , 1.        , 1.        , 0.93333333])
ic| name: 'RidgeClassifier'
    scores: array([0.86666667, 0.86666667, 0.96666667, 0.8       , 0.76666667])
ic| name: 'RidgeClassifierCV'
    scores: array([0.86666667, 0.86666667, 0.96666667, 0.8       , 0.76666667])
ic| name: 'SGDClassifier'
    scores: array([0.63333333, 0.93333333, 0.93333333, 0.93333333, 0.86666667])
ic| name: 'SVC'
    scores: array([0.93333333, 0.9       , 0.96666667, 1.        , 0.93333333])
StackingClassifier 은 오류가 나서 실행하지 않음
VotingClassifier 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_cl): 41
'''

# model.save('../_save/ml04_1_iris.h5')