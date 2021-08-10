# Machine Learning #1 [machine_learning model with iris_dataset]

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

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
ic| allAlgorithms_rg: [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
                        ('AdaBoostRegressor',
                         <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>),
                        ('BaggingRegressor', <class 'sklearn.ensemble._bagging.BaggingRegressor'>),
                        ('BayesianRidge', <class 'sklearn.linear_model._bayes.BayesianRidge'>),
                        ('CCA', <class 'sklearn.cross_decomposition._pls.CCA'>),
                        ('DecisionTreeRegressor',
                         <class 'sklearn.tree._classes.DecisionTreeRegressor'>),
                        ('DummyRegressor', <class 'sklearn.dummy.DummyRegressor'>),
                        ('ElasticNet', <class 'sklearn.linear_model._coordinate_descent.ElasticNet'>),
                        ('ElasticNetCV',
                         <class 'sklearn.linear_model._coordinate_descent.ElasticNetCV'>),
                        ('ExtraTreeRegressor', <class 'sklearn.tree._classes.ExtraTreeRegressor'>),
                        ('ExtraTreesRegressor',
                         <class 'sklearn.ensemble._forest.ExtraTreesRegressor'>),
                        ('GammaRegressor', <class 'sklearn.linear_model._glm.glm.GammaRegressor'>),
                        ('GaussianProcessRegressor',
                         <class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>),
                        ('GradientBoostingRegressor',
                         <class 'sklearn.ensemble._gb.GradientBoostingRegressor'>),
                        ('HistGradientBoostingRegressor',
                         <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor'>),
                        ('HuberRegressor', <class 'sklearn.linear_model._huber.HuberRegressor'>),
                        ('IsotonicRegression', <class 'sklearn.isotonic.IsotonicRegression'>),
                        ('KNeighborsRegressor',
                         <class 'sklearn.neighbors._regression.KNeighborsRegressor'>),
                        ('KernelRidge', <class 'sklearn.kernel_ridge.KernelRidge'>),
                        ('Lars', <class 'sklearn.linear_model._least_angle.Lars'>),
                        ('LarsCV', <class 'sklearn.linear_model._least_angle.LarsCV'>),
                        ('Lasso', <class 'sklearn.linear_model._coordinate_descent.Lasso'>),
                        ('LassoCV', <class 'sklearn.linear_model._coordinate_descent.LassoCV'>),
                        ('LassoLars', <class 'sklearn.linear_model._least_angle.LassoLars'>),
                        ('LassoLarsCV', <class 'sklearn.linear_model._least_angle.LassoLarsCV'>),
                        ('LassoLarsIC', <class 'sklearn.linear_model._least_angle.LassoLarsIC'>),
                        ('LinearRegression', <class 'sklearn.linear_model._base.LinearRegression'>),
                        ('LinearSVR', <class 'sklearn.svm._classes.LinearSVR'>),
                        ('MLPRegressor',
                         <class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>),
                        ('MultiOutputRegressor', <class 'sklearn.multioutput.MultiOutputRegressor'>),
                        ('MultiTaskElasticNet',
                         <class 'sklearn.linear_model._coordinate_descent.MultiTaskElasticNet'>),
                        ('MultiTaskElasticNetCV',
                         <class 'sklearn.linear_model._coordinate_descent.MultiTaskElasticNetCV'>),
                        ('MultiTaskLasso',
                         <class 'sklearn.linear_model._coordinate_descent.MultiTaskLasso'>),
                        ('MultiTaskLassoCV',
                         <class 'sklearn.linear_model._coordinate_descent.MultiTaskLassoCV'>),
                        ('NuSVR', <class 'sklearn.svm._classes.NuSVR'>),
                        ('OrthogonalMatchingPursuit',
                         <class 'sklearn.linear_model._omp.OrthogonalMatchingPursuit'>),
                        ('OrthogonalMatchingPursuitCV',
                         <class 'sklearn.linear_model._omp.OrthogonalMatchingPursuitCV'>),
                        ('PLSCanonical', <class 'sklearn.cross_decomposition._pls.PLSCanonical'>),
                        ('PLSRegression', <class 'sklearn.cross_decomposition._pls.PLSRegression'>),
                        ('PassiveAggressiveRegressor',
                         <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor'>),
                        ('PoissonRegressor', <class 'sklearn.linear_model._glm.glm.PoissonRegressor'>),
                        ('RANSACRegressor', <class 'sklearn.linear_model._ransac.RANSACRegressor'>),
                        ('RadiusNeighborsRegressor',
                         <class 'sklearn.neighbors._regression.RadiusNeighborsRegressor'>),
                        ('RandomForestRegressor',
                         <class 'sklearn.ensemble._forest.RandomForestRegressor'>),
                        ('RegressorChain', <class 'sklearn.multioutput.RegressorChain'>),
                        ('Ridge', <class 'sklearn.linear_model._ridge.Ridge'>),
                        ('RidgeCV', <class 'sklearn.linear_model._ridge.RidgeCV'>),
                        ('SGDRegressor',
                         <class 'sklearn.linear_model._stochastic_gradient.SGDRegressor'>),
                        ('SVR', <class 'sklearn.svm._classes.SVR'>),
                        ('StackingRegressor', <class 'sklearn.ensemble._stacking.StackingRegressor'>),
                        ('TheilSenRegressor',
                         <class 'sklearn.linear_model._theil_sen.TheilSenRegressor'>),
                        ('TransformedTargetRegressor',
                         <class 'sklearn.compose._target.TransformedTargetRegressor'>),
                        ('TweedieRegressor', <class 'sklearn.linear_model._glm.glm.TweedieRegressor'>),
                        ('VotingRegressor', <class 'sklearn.ensemble._voting.VotingRegressor'>)]
'''

for (name, algorithm) in allAlgorithms_cl:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc_score = accuracy_score(y_test, y_predict)
        ic(name, acc_score)
    except:
        print(name, '은 오류가 나서 실행하지 않음')
        continue

'''
ic| name: 'AdaBoostClassifier', acc_score: 0.9824561403508771
ic| name: 'BaggingClassifier', acc_score: 0.9210526315789473
ic| name: 'BernoulliNB', acc_score: 0.6052631578947368
ic| name: 'CalibratedClassifierCV', acc_score: 0.9649122807017544
CategoricalNB 은 오류가 나서 실행하지 않음
ClassifierChain 은 오류가 나서 실행하지 않음
ic| name: 'ComplementNB', acc_score: 0.8596491228070176
ic| name: 'DecisionTreeClassifier', acc_score: 0.9122807017543859
ic| name: 'DummyClassifier', acc_score: 0.6052631578947368
ic| name: 'ExtraTreeClassifier', acc_score: 0.9473684210526315
ic| name: 'ExtraTreesClassifier', acc_score: 0.9649122807017544
ic| name: 'GaussianNB', acc_score: 0.956140350877193
ic| name: 'GaussianProcessClassifier', acc_score: 0.9736842105263158
ic| name: 'GradientBoostingClassifier', acc_score: 0.9473684210526315
ic| name: 'HistGradientBoostingClassifier'
    acc_score: 0.956140350877193
ic| name: 'KNeighborsClassifier', acc_score: 0.9736842105263158
ic| name: 'LabelPropagation', acc_score: 0.9649122807017544
ic| name: 'LabelSpreading', acc_score: 0.9649122807017544
ic| name: 'LinearDiscriminantAnalysis', acc_score: 0.9649122807017544
ic| name: 'LinearSVC', acc_score: 0.9649122807017544
ic| name: 'LogisticRegression', acc_score: 0.9736842105263158
ic| name: 'LogisticRegressionCV', acc_score: 0.9824561403508771
ic| name: 'MLPClassifier', acc_score: 0.9649122807017544
MultiOutputClassifier 은 오류가 나서 실행하지 않음
ic| name: 'MultinomialNB', acc_score: 0.8157894736842105
ic| name: 'NearestCentroid', acc_score: 0.9649122807017544
ic| name: 'NuSVC', acc_score: 0.956140350877193
OneVsOneClassifier 은 오류가 나서 실행하지 않음
OneVsRestClassifier 은 오류가 나서 실행하지 않음
OutputCodeClassifier 은 오류가 나서 실행하지 않음
ic| name: 'PassiveAggressiveClassifier', acc_score: 0.9736842105263158
ic| name: 'Perceptron', acc_score: 0.9385964912280702
ic| name: 'QuadraticDiscriminantAnalysis'
    acc_score: 0.956140350877193
ic| name: 'RadiusNeighborsClassifier', acc_score: 0.8421052631578947
ic| name: 'RandomForestClassifier', acc_score: 0.956140350877193
ic| name: 'RidgeClassifier', acc_score: 0.9649122807017544
ic| name: 'RidgeClassifierCV', acc_score: 0.9736842105263158
ic| name: 'SGDClassifier', acc_score: 0.9649122807017544
ic| name: 'SVC', acc_score: 0.9824561403508771
StackingClassifier 은 오류가 나서 실행하지 않음
VotingClassifier 은 오류가 나서 실행하지 않음
'''

# model.save('../_save/ml04_1_iris.h5')