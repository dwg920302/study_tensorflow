# Machine Learning #6 [KFold + all_estimators]

# 그냥 4+5한 것

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# from sklearn.utils.testing import all_estimators  # (0.24 이전 버전에서)
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_diabetes()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=61)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y 카테고리화 하지 않음 (auto)

# Model

# allAlgorithms_cl = all_estimators(type_filter='classifier')   # 모든 모델
# ic(allAlgorithms_cl)
allAlgorithms_rg = all_estimators(type_filter='regressor')
# ic(allAlgorithms_rg)

kfold = KFold(n_splits=5, shuffle=True, random_state=61)

for (name, algorithm) in allAlgorithms_rg:
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

ic(len(allAlgorithms_rg))    # ic| len(allAlgorithms_cl): 41, len(allAlgorithms_rg): 54

'''
# ????? 이건 왜 잘 나옴?

ic| name: 'ARDRegression'
    scores: array([0.4335643 , 0.55437279, 0.4734882 , 0.52233512, 0.4435619 ])
ic| name: 'AdaBoostRegressor'
    scores: array([0.43782864, 0.51569112, 0.44988803, 0.42705251, 0.38169938])
ic| name: 'BaggingRegressor'
    scores: array([0.34924488, 0.4645927 , 0.36976407, 0.43828155, 0.3154123 ])
ic| name: 'BayesianRidge'
    scores: array([0.43238123, 0.55906939, 0.46633816, 0.52687268, 0.44785531])
ic| name: 'CCA'
    scores: array([0.37078059, 0.55748421, 0.46432566, 0.48188597, 0.4216686 ])
ic| name: 'DecisionTreeRegressor'
    scores: array([ 0.0256399 ,  0.09022473, -0.07488187,  0.03860948, -0.15040129])
ic| name: 'DummyRegressor'
    scores: array([-0.00453093, -0.01954984, -0.004821  , -0.00200877, -0.00309831])
ic| name: 'ElasticNet'
    scores: array([ 0.00350902, -0.01030104,  0.00313146,  0.00690496,  0.00621507])
ic| name: 'ElasticNetCV'
    scores: array([0.39341183, 0.49685943, 0.40499568, 0.47027001, 0.4269997 ])
ic| name: 'ExtraTreeRegressor'
    scores: array([-0.16017706, -0.21873786,  0.10856996, -0.1352848 , -0.18798547])
ic| name: 'ExtraTreesRegressor'
    scores: array([0.41857707, 0.49226041, 0.48738655, 0.4987729 , 0.36218435])
ic| name: 'GammaRegressor'
    scores: array([ 0.00133982, -0.01222724,  0.00091077,  0.00407917,  0.00331087])
ic| name: 'GaussianProcessRegressor'
    scores: array([-10.2479206 ,  -8.25677594, -20.51363344, -11.26158181,
                   -13.38472143])
ic| name: 'GradientBoostingRegressor'
    scores: array([0.39015009, 0.48897212, 0.44616667, 0.49603379, 0.35881869])
ic| name: 'HistGradientBoostingRegressor'
    scores: array([0.38525222, 0.48932124, 0.4394946 , 0.48510481, 0.28226911])
ic| name: 'HuberRegressor'
    scores: array([0.43434744, 0.55815096, 0.46009089, 0.49987532, 0.43199169])
ic| name: 'IsotonicRegression'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'KNeighborsRegressor'
    scores: array([0.30563541, 0.42563077, 0.38191071, 0.47689011, 0.26448165])
ic| name: 'KernelRidge'
    scores: array([-3.4241779 , -3.75209445, -3.00371627, -3.60323406, -4.12888241])
ic| name: 'Lars'
    scores: array([ 0.32434057,  0.55811475,  0.46404081,  0.52439041, -1.29446523])
ic| name: 'LarsCV'
    scores: array([0.42601654, 0.54524648, 0.47204281, 0.49816137, 0.45047848])
ic| name: 'Lasso'
    scores: array([0.31880968, 0.36333636, 0.34060595, 0.34973935, 0.33634285])
ic| name: 'LassoCV'
    scores: array([0.43217195, 0.54821865, 0.46535447, 0.52592651, 0.44545667])
ic| name: 'LassoLars'
    scores: array([0.33566229, 0.39678703, 0.3681262 , 0.3769882 , 0.3505347 ])
ic| name: 'LassoLarsCV'
    scores: array([0.43226326, 0.54524648, 0.46404081, 0.52515756, 0.43849526])
ic| name: 'LassoLarsIC'
    scores: array([0.43431329, 0.54575898, 0.47035155, 0.51071043, 0.44419725])
ic| name: 'LinearRegression'
    scores: array([0.43695487, 0.55811475, 0.46404081, 0.52439041, 0.43849526])
ic| name: 'LinearSVR'
    scores: array([-0.30268204, -0.51497359, -0.25540235, -0.43546249, -0.36185204])
ic| name: 'MLPRegressor'
    scores: array([-2.89396467, -3.19720825, -2.683187  , -2.93985845, -3.25853037])
MultiOutputRegressor 은 오류가 나서 실행하지 않음
ic| name: 'MultiTaskElasticNet'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'MultiTaskElasticNetCV'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'MultiTaskLasso', scores: array([nan, nan, nan, nan, nan])
ic| name: 'MultiTaskLassoCV', scores: array([nan, nan, nan, nan, nan])
ic| name: 'NuSVR'
    scores: array([0.1666491 , 0.15758559, 0.14032128, 0.15578816, 0.1739619 ])
ic| name: 'OrthogonalMatchingPursuit'
    scores: array([0.2284861 , 0.36694109, 0.23454564, 0.23450149, 0.28318935])
ic| name: 'OrthogonalMatchingPursuitCV'
    scores: array([0.40439938, 0.54772207, 0.47369662, 0.51733546, 0.46593336])
ic| name: 'PLSCanonical'
    scores: array([-0.90275819, -1.26960762, -0.93640687, -1.03459756, -1.93005501])
ic| name: 'PLSRegression'
    scores: array([0.4202881 , 0.56272379, 0.46028112, 0.5533018 , 0.44420237])
ic| name: 'PassiveAggressiveRegressor'
    scores: array([0.43569272, 0.50941586, 0.44826402, 0.519655  , 0.41710138])
ic| name: 'PoissonRegressor'
    scores: array([0.3009424 , 0.38463664, 0.2946976 , 0.3542471 , 0.33161758])
ic| name: 'RANSACRegressor'
    scores: array([-0.54571885,  0.2111447 , -1.22875609, -0.3767865 , -0.37696582])
ic| name: 'RadiusNeighborsRegressor'
    scores: array([-0.00453093, -0.01954984, -0.004821  , -0.00200877, -0.00309831])
ic| name: 'RandomForestRegressor'
    scores: array([0.42012988, 0.50718612, 0.43816066, 0.46892474, 0.36276746])
RegressorChain 은 오류가 나서 실행하지 않음
ic| name: 'Ridge'
    scores: array([0.37903713, 0.47328688, 0.38564278, 0.44779172, 0.41465913])
ic| name: 'RidgeCV'
    scores: array([0.43377143, 0.55618358, 0.46271332, 0.52939212, 0.45397749])
ic| name: 'SGDRegressor'
    scores: array([0.37102618, 0.46006291, 0.37279911, 0.43241682, 0.39722988])
ic| name: 'SVR'
    scores: array([0.18635907, 0.13201397, 0.15217333, 0.14970397, 0.17476955])
StackingRegressor 은 오류가 나서 실행하지 않음
ic| name: 'TheilSenRegressor'
    scores: array([0.43012386, 0.55287221, 0.4628098 , 0.48111527, 0.42204099])
ic| name: 'TransformedTargetRegressor'
    scores: array([0.43695487, 0.55811475, 0.46404081, 0.52439041, 0.43849526])
ic| name: 'TweedieRegressor'
    scores: array([ 0.00140053, -0.01246286,  0.00100289,  0.00452697,  0.00378242])
VotingRegressor 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_rg): 54
'''

# model.save('../_save/ml04_1_iris.h5')