# 실습
# Feature Importance(줄여서 FI)가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋을 재구성 후
# 각 모델별로 결과 도출, 기존 모델과 비교
# DataFrame

from icecream import ic
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # Tree의 확장형


dataset = load_iris()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=221)

# to_Dataframe
x_train = pd.DataFrame(x_train, columns=dataset.feature_names)
x_test = pd.DataFrame(x_test, columns=dataset.feature_names)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Model

# model = DecisionTreeClassifier()
model = RandomForestClassifier()

model.fit(x_train, y_train)

ic(model.score(x_test, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
ic(model.feature_importances_)
# ic(dataset.feature_names)

# here
percentage = 20 / 100
size = x_train.shape[1]
res = np.sort(model.feature_importances_)[:round(size * percentage)]

# ic(size, res)

label = pd.DataFrame(np.array([model.feature_importances_]), columns=dataset.feature_names)

# ic(label)

label = label.transpose()

# ic(label)

label = label.sort_values(by=0)

label.columns = ['value']

# ic(label)

got_columns = label[:round(size * percentage)].index
ic(got_columns)

x_train_tmp = x_train.copy()
x_test_tmp = x_test.copy()

# ic(x_train_tmp.shape, x_test_tmp.shape)

# ic(x_train_tmp.head(), x_test_tmp.tail())

x_train_tmp = x_train_tmp.drop(got_columns, axis=1)
x_test_tmp = x_test_tmp.drop(got_columns, axis=1)

# ic(x_train_tmp.shape, x_test_tmp.shape)
ic(x_train_tmp.head(), x_test_tmp.tail())

model.fit(x_train_tmp, y_train)

ic(model.score(x_test_tmp, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test_tmp)
print('정답률 : ', accuracy_score(y_test, y_predict))
ic(model.feature_importances_)

'''
DecisionTreeClassifier
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.03546839, 0.01251826, 0.05471655, 0.89729681])

(with less columns) 4 -> 3
ic| model.score(x_test_tmp, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.04729119, 0.0672348 , 0.88547401])

RandomForestClassifier
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.12440901, 0.02375816, 0.44381092, 0.40802191])

ic| model.score(x_test_tmp, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.20745665, 0.41460323, 0.37794012])
-> 칼럼이 준 거 빼고는 유의미한 차이가 보이지 않음
'''