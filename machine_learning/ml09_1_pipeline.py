from icecream import ic
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline


dataset = load_iris()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},  # 4 * 1 * 5 = 20
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},  # 3 * 1 * 2 * 5 = 30
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} #   4 * 1 * 2 * 5 = 40
]   # 총 20+30+40 = 90번 연산. 5는 kfold의 split
# degree?

# pipeline 만들기

model = make_pipeline(MinMaxScaler(), SVC())

model.fit(x_train, y_train)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
'''
ic| model.score(x_test, y_test): 1.0
정답률 :  1.0
'''