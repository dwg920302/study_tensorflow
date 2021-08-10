
# 회귀 (Regressor) 데이터를 Classifier로 만들었을 경우에 에러 확인 -> ValueError: Unknown label type: 'continuous'

from icecream import ic

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # 분류 모델 (회귀 아님. 절대.)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import r2_score
# svm -> Support Vector Machine


dataset = load_boston()

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

# y 카테고리화 하지 않음 (auto)

# Model

# model = LinearSVC()   # ValueError: Unknown label type: 'continuous'
# 안 됨

# model = SVC()   # ValueError: Unknown label type: 'continuous'
# 안 됨

# model = KNeighborsClassifier() # ValueError: Unknown label type: 'continuous'
# 안 됨

# model = KNeighborsRegressor()   # ValueError: continuous is not supported -> accuracy_score를 r2로 바꿔주면 해결
# accuracy :  0.8061083911657948

# model = LogisticRegression()    # ValueError: Unknown label type: 'continuous'
# Regression이라고 써있지만 사실상 Classifier와 같음. Classifier와 같은 에러 발생

# model = LinearRegression()
# r2_score :  0.7057189451172973

# model = DecisionTreeRegressor()
# r2_score :  0.7575622318419761

model = RandomForestRegressor()
# r2_score :  0.9092125886306707
# r2_score :  0.9017375319241426

# compile 없음
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('accuracy : ', results)   # accuracy
# loss = model.evaluate(x_test, y_test)

# print('loss = ', loss[0], ', accuracy = ', loss[1])

y_pred = model.predict(x_test)

ic(y_pred, y_pred.shape, y_test, y_test.shape)

r2_score = r2_score(y_test, y_pred)
print('r2_score : ', r2_score)

ic(y_pred)