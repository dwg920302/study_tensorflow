# Machine Learning #1 [machine_learning model with iris_dataset]

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_diabetes()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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

for (name, algorithm) in allAlgorithms_rg:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        r2_score = r2_score(y_test, y_predict)
        ic(name, r2_score)
    except:
        print(name, '은 오류가 나서 실행하지 않음')
        continue

'''
ic| name: 'ARDRegression', r2_score: 0.40711326851673446
AdaBoostRegressor 은 오류가 나서 실행하지 않음
BaggingRegressor 은 오류가 나서 실행하지 않음
BayesianRidge 은 오류가 나서 실행하지 않음
CCA 은 오류가 나서 실행하지 않음
DecisionTreeRegressor 은 오류가 나서 실행하지 않음
DummyRegressor 은 오류가 나서 실행하지 않음
ElasticNet 은 오류가 나서 실행하지 않음
ElasticNetCV 은 오류가 나서 실행하지 않음
ExtraTreeRegressor 은 오류가 나서 실행하지 않음
ExtraTreesRegressor 은 오류가 나서 실행하지 않음
GammaRegressor 은 오류가 나서 실행하지 않음
GaussianProcessRegressor 은 오류가 나서 실행하지 않음
GradientBoostingRegressor 은 오류가 나서 실행하지 않음
HistGradientBoostingRegressor 은 오류가 나서 실행하지 않음
HuberRegressor 은 오류가 나서 실행하지 않음
IsotonicRegression 은 오류가 나서 실행하지 않음
KNeighborsRegressor 은 오류가 나서 실행하지 않음
KernelRidge 은 오류가 나서 실행하지 않음
Lars 은 오류가 나서 실행하지 않음
LarsCV 은 오류가 나서 실행하지 않음
Lasso 은 오류가 나서 실행하지 않음
LassoCV 은 오류가 나서 실행하지 않음
LassoLars 은 오류가 나서 실행하지 않음
LassoLarsCV 은 오류가 나서 실행하지 않음
LassoLarsIC 은 오류가 나서 실행하지 않음
LinearRegression 은 오류가 나서 실행하지 않음
LinearSVR 은 오류가 나서 실행하지 않음
MLPRegressor 은 오류가 나서 실행하지 않음
MultiOutputRegressor 은 오류가 나서 실행하지 않음
MultiTaskElasticNet 은 오류가 나서 실행하지 않음
MultiTaskElasticNetCV 은 오류가 나서 실행하지 않음
MultiTaskLasso 은 오류가 나서 실행하지 않음
MultiTaskLassoCV 은 오류가 나서 실행하지 않음
NuSVR 은 오류가 나서 실행하지 않음
OrthogonalMatchingPursuit 은 오류가 나서 실행하지 않음
OrthogonalMatchingPursuitCV 은 오류가 나서 실행하지 않음
PLSCanonical 은 오류가 나서 실행하지 않음
PLSRegression 은 오류가 나서 실행하지 않음
PassiveAggressiveRegressor 은 오류가 나서 실행하지 않음
PoissonRegressor 은 오류가 나서 실행하지 않음
RANSACRegressor 은 오류가 나서 실행하지 않음
RadiusNeighborsRegressor 은 오류가 나서 실행하지 않음
RandomForestRegressor 은 오류가 나서 실행하지 않음
RegressorChain 은 오류가 나서 실행하지 않음
Ridge 은 오류가 나서 실행하지 않음
RidgeCV 은 오류가 나서 실행하지 않음
SGDRegressor 은 오류가 나서 실행하지 않음
SVR 은 오류가 나서 실행하지 않음
StackingRegressor 은 오류가 나서 실행하지 않음
TheilSenRegressor 은 오류가 나서 실행하지 않음
TransformedTargetRegressor 은 오류가 나서 실행하지 않음
TweedieRegressor 은 오류가 나서 실행하지 않음
VotingRegressor 은 오류가 나서 실행하지 않음
'''

# model.save('../_save/ml04_1_iris.h5')