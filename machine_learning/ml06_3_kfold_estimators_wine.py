# Machine Learning #6 [KFold + all_estimators]

# 그냥 4+5한 것

from icecream import ic

import warnings

import numpy as np
import pandas as pd
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


dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality'].to_numpy()
x = dataset.drop(columns='quality')

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
    scores: array([0.45510204, 0.35408163, 0.43367347, 0.45250255, 0.41368744])
ic| name: 'BaggingClassifier'
    scores: array([0.65      , 0.62142857, 0.65612245, 0.66496425, 0.64351379])
ic| name: 'BernoulliNB'
    scores: array([0.41530612, 0.46938776, 0.45306122, 0.4433095 , 0.46271706])
ic| name: 'CalibratedClassifierCV'
    scores: array([0.47040816, 0.50102041, 0.49387755, 0.49233912, 0.51481103])
ic| name: 'CategoricalNB'
    scores: array([0.49693878, 0.49081633,        nan,        nan,        nan])
ClassifierChain 은 오류가 나서 실행하지 않음
ic| name: 'ComplementNB'
    scores: array([0.38469388, 0.35102041, 0.35714286, 0.37589377, 0.34729316])
ic| name: 'DecisionTreeClassifier'
    scores: array([0.59795918, 0.6255102 , 0.59489796, 0.61491318, 0.59141982])
ic| name: 'DummyClassifier'
    scores: array([0.41530612, 0.46938776, 0.45306122, 0.4433095 , 0.46271706])
ic| name: 'ExtraTreeClassifier'
    scores: array([0.60714286, 0.59693878, 0.6244898 , 0.60367722, 0.57814096])
ic| name: 'ExtraTreesClassifier'
    scores: array([0.68979592, 0.68469388, 0.70714286, 0.69458631, 0.69662921])
ic| name: 'GaussianNB'
    scores: array([0.45510204, 0.43877551, 0.42142857, 0.46373851, 0.4412666 ])
ic| name: 'GaussianProcessClassifier'
    scores: array([0.58877551, 0.58163265, 0.58877551, 0.59346272, 0.59346272])
ic| name: 'GradientBoostingClassifier'
    scores: array([0.58979592, 0.61428571, 0.57040816, 0.59754852, 0.60163432])
ic| name: 'HistGradientBoostingClassifier'
    scores: array([0.68469388, 0.67959184, 0.67244898, 0.6680286 , 0.68845761])
ic| name: 'KNeighborsClassifier'
    scores: array([0.47142857, 0.4622449 , 0.48163265, 0.50766088, 0.47191011])
ic| name: 'LabelPropagation'
    scores: array([0.59081633, 0.57040816, 0.57755102, 0.58222676, 0.57711951])
ic| name: 'LabelSpreading'
    scores: array([0.59081633, 0.57142857, 0.57755102, 0.58222676, 0.57711951])
ic| name: 'LinearDiscriminantAnalysis'
    scores: array([0.51938776, 0.53061224, 0.51020408, 0.53932584, 0.53932584])
ic| name: 'LinearSVC'
    scores: array([0.37857143, 0.47142857, 0.31428571, 0.4351379 , 0.46680286])
ic| name: 'LogisticRegression'
    scores: array([0.46326531, 0.47244898, 0.46938776, 0.47395301, 0.46475996])
ic| name: 'LogisticRegressionCV'
    scores: array([0.4755102 , 0.50714286, 0.49489796, 0.52196118, 0.50561798])
ic| name: 'MLPClassifier'
    scores: array([0.45612245, 0.5       , 0.47142857, 0.51481103, 0.50970378])
MultiOutputClassifier 은 오류가 나서 실행하지 않음
ic| name: 'MultinomialNB'
    scores: array([0.3877551 , 0.38877551, 0.41428571, 0.38712972, 0.38610827])
ic| name: 'NearestCentroid'
    scores: array([0.11632653, 0.11122449, 0.26326531, 0.10623085, 0.10418795])
ic| name: 'NuSVC', scores: array([nan, nan, nan, nan, nan])
OneVsOneClassifier 은 오류가 나서 실행하지 않음
OneVsRestClassifier 은 오류가 나서 실행하지 않음
OutputCodeClassifier 은 오류가 나서 실행하지 않음
ic| name: 'PassiveAggressiveClassifier'
    scores: array([0.32142857, 0.33877551, 0.45306122, 0.31256384, 0.29111338])
ic| name: 'Perceptron'
    scores: array([0.41734694, 0.18571429, 0.26836735, 0.37385087, 0.31052094])
ic| name: 'QuadraticDiscriminantAnalysis'
    scores: array([0.50714286, 0.45612245, 0.48163265, 0.49540347, 0.47191011])
ic| name: 'RadiusNeighborsClassifier'
    scores: array([nan, nan, nan, nan, nan])
ic| name: 'RandomForestClassifier'
    scores: array([0.67040816, 0.6877551 , 0.69693878, 0.68641471, 0.69050051])
ic| name: 'RidgeClassifier'
    scores: array([0.51530612, 0.52653061, 0.52755102, 0.53115424, 0.52706844])
ic| name: 'RidgeClassifierCV'
    scores: array([0.51530612, 0.52653061, 0.52755102, 0.53115424, 0.52706844])
ic| name: 'SGDClassifier'
    scores: array([0.41734694, 0.47959184, 0.17755102, 0.49540347, 0.48416752])
ic| name: 'SVC'
    scores: array([0.41428571, 0.45816327, 0.45714286, 0.44228805, 0.46475996])
StackingClassifier 은 오류가 나서 실행하지 않음
VotingClassifier 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_cl): 41
'''

# model.save('../_save/ml04_1_iris.h5')