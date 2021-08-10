

from icecream import ic

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     # 분류 모델 (회귀 아님. 절대.)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score, accuracy_score
# svm -> Support Vector Machine


dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=12)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y 카테고리화 하지 않음 (auto)

# Model

model = LinearSVC()
# accuracy :  0.9649122807017544
# accuracy_score :  0.9649122807017544

# model = SVC()
# accuracy :  0.956140350877193
# accuracy_score :  0.956140350877193

# model = KNeighborsClassifier()
# accuracy :  0.956140350877193
# accuracy_score :  0.956140350877193

# model = LogisticRegression()
# accuracy :  0.9473684210526315
# accuracy_score :  0.9473684210526315

# model = DecisionTreeClassifier()
# accuracy :  0.9210526315789473
# accuracy_score :  0.9210526315789473

# model = RandomForestClassifier()
# accuracy :  0.9298245614035088
# accuracy_score :  0.9298245614035088

# compile 없음
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('accuracy : ', results)   # accuracy
# loss = model.evaluate(x_test, y_test)

# print('loss = ', loss[0], ', accuracy = ', loss[1])

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy_score : ', acc)

ic(y_pred)