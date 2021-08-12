from icecream import ic
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
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

parameters =[
    {'rf__min_samples_leaf' : [3, 5, 7],
    'rf__max_depth' : [2, 3, 5, 10],
    'rf__min_samples_split' : [6, 8, 10]}
]   # 이름을 맞춰줘야 함.

# pipeline 만들기 (with Pipeline)

pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestClassifier())])

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

model.fit(x_train, y_train)

ic(model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
'''
Fitting 5 folds for each of 36 candidates, totalling 180 fits
ic| model.score(x_test, y_test): 1.0
정답률 :  1.0
'''