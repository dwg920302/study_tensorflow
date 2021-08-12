# Machine Learning #1 [machine_learning model with iris_dataset]

from icecream import ic

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

from sklearn.metrics import r2_score

# from sklearn.utils.testing import all_estimators  # (0.24 이전 버전에서)
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_boston()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

# x = np.load('../_save/_npy/k55_x_data_boston.npy')
# y = np.load('../_save/_npy/k55_y_data_boston.npy')

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=32)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# y 카테고리화 하지 않음 (auto)

# Model

# allAlgorithms_cl = all_estimators(type_filter='classifier')   # 모든 모델
# ic(allAlgorithms_cl)
allAlgorithms_rg = all_estimators(type_filter='regressor')
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

# y_test = y_test.astype(np.float)

# Regressor를 반복 사용 시에는 Classifier와 달리 r2_score를 구하는 과정에서 자꾸 float64 에러가 발생함

for (name, algorithm) in allAlgorithms_rg:
    # x_train_copy = x_train.copy()
    # x_test_copy = x_test.copy()
    # y_train_copy = y_train.copy()
    # y_test_copy = y_test.copy()

    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        ic('Error Point_1 float64 piece of shit')
        r2_score = r2_score(y_test, y_predict)  # here error fucking float64
        ic('Error Point_2')
        ic(name, r2_score)
    except Exception as e:
        ic(e)
        print(name, '은 오류가 나서 실행하지 않음')
        continue

ic(len(allAlgorithms_rg))

'''
-> 현쨰 먠 처음 하나 빼고 전부 TypeError:"'numpy.float64' object is not callable" 에러 때문에 안되고 있음.
-> 형변환 -> 해도 안 됨

ic| 1
ic| 2
ic| name: 'ARDRegression', r2_score: 0.7022916883315944
ic| 1
AdaBoostRegressor 은 오류가 나서 실행하지 않음
ic| 1
BaggingRegressor 은 오류가 나서 실행하지 않음
ic| 1
BayesianRidge 은 오류가 나서 실행하지 않음
ic| 1
CCA 은 오류가 나서 실행하지 않음
ic| 1
DecisionTreeRegressor 은 오류가 나서 실행하지 않음
ic| 1
DummyRegressor 은 오류가 나서 실행하지 않음
ic| 1
ElasticNet 은 오류가 나서 실행하지 않음
ic| 1
ElasticNetCV 은 오류가 나서 실행하지 않음
ic| 1
ExtraTreeRegressor 은 오류가 나서 실행하지 않음
ic| 1
ExtraTreesRegressor 은 오류가 나서 실행하지 않음
ic| 1
GammaRegressor 은 오류가 나서 실행하지 않음
ic| 1
GaussianProcessRegressor 은 오류가 나서 실행하지 않음
ic| 1
GradientBoostingRegressor 은 오류가 나서 실행하지 않음
ic| 1
HistGradientBoostingRegressor 은 오류가 나서 실행하지 않음
ic| 1
HuberRegressor 은 오류가 나서 실행하지 않음
IsotonicRegression 은 오류가 나서 실행하지 않음
ic| 1
KNeighborsRegressor 은 오류가 나서 실행하지 않음
ic| 1
KernelRidge 은 오류가 나서 실행하지 않음
ic| 1
Lars 은 오류가 나서 실행하지 않음
ic| 1
LarsCV 은 오류가 나서 실행하지 않음
ic| 1
Lasso 은 오류가 나서 실행하지 않음
ic| 1
LassoCV 은 오류가 나서 실행하지 않음
ic| 1
LassoLars 은 오류가 나서 실행하지 않음
ic| 1
LassoLarsCV 은 오류가 나서 실행하지 않음
ic| 1
LassoLarsIC 은 오류가 나서 실행하지 않음
ic| 1
LinearRegression 은 오류가 나서 실행하지 않음
ic| 1
LinearSVR 은 오류가 나서 실행하지 않음
ic| 1
MLPRegressor 은 오류가 나서 실행하지 않음
MultiOutputRegressor 은 오류가 나서 실행하지 않음
MultiTaskElasticNet 은 오류가 나서 실행하지 않음
MultiTaskElasticNetCV 은 오류가 나서 실행하지 않음
MultiTaskLasso 은 오류가 나서 실행하지 않음
MultiTaskLassoCV 은 오류가 나서 실행하지 않음
ic| 1
NuSVR 은 오류가 나서 실행하지 않음
ic| 1
OrthogonalMatchingPursuit 은 오류가 나서 실행하지 않음
ic| 1
OrthogonalMatchingPursuitCV 은 오류가 나서 실행하지 않음
ic| 1
PLSCanonical 은 오류가 나서 실행하지 않음
ic| 1
PLSRegression 은 오류가 나서 실행하지 않음
ic| 1
PassiveAggressiveRegressor 은 오류가 나서 실행하지 않음
ic| 1
PoissonRegressor 은 오류가 나서 실행하지 않음
ic| 1
RANSACRegressor 은 오류가 나서 실행하지 않음
ic| 1
RadiusNeighborsRegressor 은 오류가 나서 실행하지 않음
ic| 1
RandomForestRegressor 은 오류가 나서 실행하지 않음
RegressorChain 은 오류가 나서 실행하지 않음
ic| 1
Ridge 은 오류가 나서 실행하지 않음
ic| 1
RidgeCV 은 오류가 나서 실행하지 않음
ic| 1
SGDRegressor 은 오류가 나서 실행하지 않음
ic| 1
SVR 은 오류가 나서 실행하지 않음
StackingRegressor 은 오류가 나서 실행하지 않음
ic| 1
TheilSenRegressor 은 오류가 나서 실행하지 않음
ic| 1
TransformedTargetRegressor 은 오류가 나서 실행하지 않음
ic| 1
TweedieRegressor 은 오류가 나서 실행하지 않음
VotingRegressor 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_rg): 54
'''

# model.save('../_save/ml04_1_iris.h5')