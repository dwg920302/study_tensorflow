# Machine Learning #1 [machine_learning model with iris_dataset]

from icecream import ic

import warnings

import numpy as np
from sklearn.datasets import load_iris
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


dataset = load_iris()

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

for (name, algorithm) in allAlgorithms_cl:
    try:
        model = algorithm()
        
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc_score = accuracy_score(y_test, y_predict)
        ic(name, acc_score)
    except Exception as e:
        ic(e)
        print(name, '은 오류가 나서 실행하지 않음')
        continue

ic(len(allAlgorithms_cl))    # ic| len(allAlgorithms_cl): 41, len(allAlgorithms_rg): 54

'''
ic| name: 'AdaBoostClassifier', acc_score: 0.9
ic| name: 'BaggingClassifier', acc_score: 0.9333333333333333
ic| name: 'BernoulliNB', acc_score: 0.23333333333333334
ic| name: 'CalibratedClassifierCV', acc_score: 0.9
ic| name: 'CategoricalNB', acc_score: 0.23333333333333334
ClassifierChain 은 오류가 나서 실행하지 않음
ic| name: 'ComplementNB', acc_score: 0.6333333333333333
ic| name: 'DecisionTreeClassifier', acc_score: 0.9
ic| name: 'DummyClassifier', acc_score: 0.23333333333333334
ic| name: 'ExtraTreeClassifier', acc_score: 0.9
ic| name: 'ExtraTreesClassifier', acc_score: 0.9
ic| name: 'GaussianNB', acc_score: 0.9
ic| name: 'GaussianProcessClassifier', acc_score: 0.9
ic| name: 'GradientBoostingClassifier', acc_score: 0.9333333333333333
ic| name: 'HistGradientBoostingClassifier', acc_score: 0.9
ic| name: 'KNeighborsClassifier', acc_score: 0.9333333333333333
ic| name: 'LabelPropagation', acc_score: 0.9
ic| name: 'LabelSpreading', acc_score: 0.9
ic| name: 'LinearDiscriminantAnalysis', acc_score: 0.9666666666666667
ic| name: 'LinearSVC', acc_score: 0.9333333333333333
ic| name: 'LogisticRegression', acc_score: 0.9
ic| name: 'LogisticRegressionCV', acc_score: 0.9333333333333333
ic| name: 'MLPClassifier', acc_score: 0.9
MultiOutputClassifier 은 오류가 나서 실행하지 않음
ic| name: 'MultinomialNB', acc_score: 0.6333333333333333
ic| name: 'NearestCentroid', acc_score: 0.9
ic| name: 'NuSVC', acc_score: 0.9
OneVsOneClassifier 은 오류가 나서 실행하지 않음
OneVsRestClassifier 은 오류가 나서 실행하지 않음
OutputCodeClassifier 은 오류가 나서 실행하지 않음
ic| name: 'PassiveAggressiveClassifier', acc_score: 0.8666666666666667
ic| name: 'Perceptron', acc_score: 0.8333333333333334
ic| name: 'QuadraticDiscriminantAnalysis'
    acc_score: 0.9666666666666667
ic| name: 'RadiusNeighborsClassifier', acc_score: 0.36666666666666664
ic| name: 'RandomForestClassifier', acc_score: 0.8666666666666667
ic| name: 'RidgeClassifier', acc_score: 0.8333333333333334
ic| name: 'RidgeClassifierCV', acc_score: 0.8333333333333334
ic| name: 'SGDClassifier', acc_score: 0.9333333333333333
ic| name: 'SVC', acc_score: 0.9
StackingClassifier 은 오류가 나서 실행하지 않음
VotingClassifier 은 오류가 나서 실행하지 않음
ic| len(allAlgorithms_cl): 41
'''

# model.save('../_save/ml04_1_iris.h5')