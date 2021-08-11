# Machine Learning #6 [KFold + all_estimators]

# 그냥 4+5한 것

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# from sklearn.utils.testing import all_estimators  # (0.24 이전 버전에서)
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_boston()

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
    scores: array([0.78900312, 0.60175729, 0.80008548, 0.77022145, 0.51980762])
ic| name: 'AdaBoostRegressor'
    scores: array([0.85978169, 0.80461081, 0.83082917, 0.84545858, 0.81922464])
ic| name: 'BaggingRegressor'
    scores: array([0.87662629, 0.8550016 , 0.86282949, 0.8476257 , 0.83095349])
ic| name: 'BayesianRidge'
    scores: array([0.77332447, 0.61294674, 0.79071838, 0.75960471, 0.56435939])
ic| name: 'CCA'
    scores: array([0.80725952, 0.51581474, 0.77151214, 0.73534022, 0.42529399])
ic| name: 'DecisionTreeRegressor'
    scores: array([0.73354356, 0.79277943, 0.76764337, 0.73695613, 0.66556111])
ic| name: 'DummyRegressor'
    scores: array([-1.85856495e-04, -3.85593117e-05, -1.37204727e-02, -9.31105472e-04,
                   -3.77080127e-03])
ic| name: 'ElasticNet'
    scores: array([0.70567822, 0.59104402, 0.75104663, 0.70522537, 0.56507561])
ic| name: 'ElasticNetCV'
    scores: array([0.68883922, 0.58665451, 0.73647709, 0.68500901, 0.55047822])
ic| name: 'ExtraTreeRegressor'
    scores: array([0.83609719, 0.66841412, 0.71855807, 0.71346239, 0.57596142])
ic| name: 'ExtraTreesRegressor'
    scores: array([0.91110281, 0.86044692, 0.90802092, 0.88473176, 0.9082952 ])
ic| name: 'GammaRegressor'
    scores: array([-1.89739729e-04, -3.93261096e-05, -1.31968103e-02, -1.02410045e-03,
                   -4.06949839e-03])
ic| name: 'GaussianProcessRegressor'
    scores: array([-4.45913785, -6.06736674, -6.29618553, -6.36853963, -7.13792248])
ic| name: 'GradientBoostingRegressor'
    scores: array([0.89919702, 0.84624659, 0.88786773, 0.87541653, 0.89973038])
ic| name: 'HistGradientBoostingRegressor'
    scores: array([0.90900983, 0.85587438, 0.89606383, 0.85479445, 0.84462751])
ic| name: 'HuberRegressor'
    scores: array([0.73599885, 0.58874344, 0.66778908, 0.72807188, 0.49868852])
ic| name: 'IsotonicRegression'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'KNeighborsRegressor'
    scores: array([0.52484459, 0.47359838, 0.51641449, 0.52917182, 0.48452546])
ic| name: 'KernelRidge'
    scores: array([0.77183003, 0.60603395, 0.76907229, 0.74354097, 0.50301463])
ic| name: 'Lars'
    scores: array([0.76031622, 0.62447007, 0.79776236, 0.77745328, 0.52097252])
ic| name: 'LarsCV'
    scores: array([0.77674189, 0.62211721, 0.79980123, 0.72758279, 0.52213686])
ic| name: 'Lasso'
    scores: array([0.6896943 , 0.57669031, 0.7520489 , 0.70012352, 0.55848509])
ic| name: 'LassoCV'
    scores: array([0.64425539, 0.60247776, 0.71130817, 0.69337626, 0.56545129])
ic| name: 'LassoLars'
    scores: array([-1.85856495e-04, -3.85593117e-05, -1.37204727e-02, -9.31105472e-04,
                   -3.77080127e-03])
ic| name: 'LassoLarsCV'
    scores: array([0.79436164, 0.62211721, 0.79980123, 0.72758279, 0.51700989])
ic| name: 'LassoLarsIC'
    scores: array([0.79246386, 0.62426612, 0.80118275, 0.77741126, 0.54641656])
ic| name: 'LinearRegression'
    scores: array([0.79502099, 0.62447007, 0.79935857, 0.77745328, 0.55092964])
ic| name: 'LinearSVR'
    scores: array([ 0.6068304 ,  0.28159407,  0.67901293,  0.51634067, -0.01392693])
ic| name: 'MLPRegressor'
    scores: array([0.45605476, 0.40547115, 0.63352535, 0.46865507, 0.44524932])
MultiOutputRegressor 은 오류가 나서 실행하지 않음
ic| name: 'MultiTaskElasticNet'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'MultiTaskElasticNetCV'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'MultiTaskLasso', scores: array([nan, nan, nan, nan, nan])
ic| name: 'MultiTaskLassoCV', scores: array([nan, nan, nan, nan, nan])
ic| name: 'NuSVR'
    scores: array([0.22451993, 0.27210804, 0.29106781, 0.20675845, 0.16775988])
ic| name: 'OrthogonalMatchingPursuit'
    scores: array([0.57900523, 0.51370853, 0.61734102, 0.54492262, 0.40323528])
ic| name: 'OrthogonalMatchingPursuitCV'
    scores: array([0.75394366, 0.58498279, 0.76843835, 0.71634252, 0.50029369])
ic| name: 'PLSCanonical'
    scores: array([-1.49821574, -2.39051544, -2.2552689 , -1.6064945 , -3.31541712])
ic| name: 'PLSRegression'
    scores: array([0.7825799 , 0.60962178, 0.77315816, 0.73531932, 0.5157384 ])
ic| name: 'PassiveAggressiveRegressor'
    scores: array([-0.58650796, -0.45181257,  0.329902  , -0.00714345,  0.05111577])
ic| name: 'PoissonRegressor'
    scores: array([0.83383866, 0.68353796, 0.8044605 , 0.77836053, 0.65062299])
ic| name: 'RANSACRegressor'
    scores: array([0.56759663, 0.58979814, 0.6033047 , 0.64627956, 0.0941429 ])
ic| name: 'RadiusNeighborsRegressor'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'RandomForestRegressor'
    scores: array([0.88911108, 0.84951079, 0.89717272, 0.86584432, 0.85915609])
RegressorChain 은 오류가 나서 실행하지 않음
ic| name: 'Ridge'
    scores: array([0.78884968, 0.61968568, 0.79821328, 0.77259881, 0.55864271])
ic| name: 'RidgeCV'
    scores: array([0.79419032, 0.62397596, 0.79962879, 0.77702527, 0.55262816])
ic| name: 'SGDRegressor'
    scores: array([-6.95415226e+24, -4.02859206e+26, -5.93128043e+26, -6.04215925e+26,
                   -1.33577972e+25])
ic| name: 'SVR'
    scores: array([0.18620424, 0.24358289, 0.27092166, 0.18027444, 0.11314424])
StackingRegressor 은 오류가 나서 실행하지 않음
ic| name: 'TheilSenRegressor'
    scores: array([0.78414166, 0.63536555, 0.76480186, 0.74734001, 0.46459768])
ic| name: 'TransformedTargetRegressor'
    scores: array([0.79502099, 0.62447007, 0.79935857, 0.77745328, 0.55092964])
ic| name: 'TweedieRegressor'
    scores: array([0.68791701, 0.60767797, 0.7179366 , 0.68533342, 0.54321659])
VotingRegressor 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_rg): 54
'''

# model.save('../_save/ml04_1_iris.h5')