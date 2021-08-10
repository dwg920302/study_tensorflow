# Machine Learning #1 [machine_learning model with iris_dataset]

from icecream import ic

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     # 분류 모델 (회귀 아님. 절대.)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score, accuracy_score
# svm -> Support Vector Machine


dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality'].to_numpy()
x = dataset.drop(columns='quality')

# Preprocessing

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y 카테고리화 하지 않음 (auto)

# Model

model = LinearSVC()
# accuracy :  0.5163265306122449
# accuracy_score :  0.5163265306122449

# model = SVC()
# accuracy :  0.5469387755102041
# accuracy_score :  0.5469387755102041

# model = KNeighborsClassifier()
# accuracy :  0.5642857142857143
# accuracy_score :  0.5642857142857143

# model = LogisticRegression()
# accuracy :  0.5244897959183673
# accuracy_score :  0.5244897959183673

# model = DecisionTreeClassifier()
# accuracy :  0.6377551020408163
# accuracy_score :  0.6377551020408163

# model = RandomForestClassifier()
# accuracy :  0.7030612244897959
# accuracy_score :  0.7030612244897959

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