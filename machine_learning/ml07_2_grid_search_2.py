# Machine Learning #5 [K-Fold]

from icecream import ic
import warnings
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# KFold, Cross_Validation

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')   # 오류구문 무시


dataset = load_iris()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

model = SVC(C=1, kernel='linear')   # 1에서 나온 최적의 estimator를 가져옴

model.fit(x_train, y_train)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

'''
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
'''


model = SVC(C=100, kernel='sigmoid', gamma=0.0001)  # 최적이 아닌 엄한 estimator를 가져옴

model.fit(x_train, y_train)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

'''
ic| model.score(x_test, y_test): 0.8666666666666667
정답률 :  0.8666666666666667
'''


model = SVC(C=10, kernel='rbf', gamma=0.0001)  # 최적델이 아닌 엄한 estimator를 가져옴

model.fit(x_train, y_train)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

'''
ic| model.score(x_test, y_test): 0.5333333333333333
정답률 :  0.5333333333333333
'''