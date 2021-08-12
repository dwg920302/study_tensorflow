# 실습
# Feature Importance(줄여서 FI)가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋을 재구성 후
# 각 모델별로 결과 도출, 기존 모델과 비교
# DataFrame

from icecream import ic
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # Tree의 확장형


dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality'].to_numpy()
x = dataset.drop(columns='quality')

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# to_Dataframe
x_train = pd.DataFrame(x_train, columns=x.columns)
x_test = pd.DataFrame(x_test, columns=x.columns)
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
# ic(x.columns)

# here
percentage = 20 / 100
size = x_train.shape[1]
res = np.sort(model.feature_importances_)[:round(size * percentage)]

# ic(size, res)

label = pd.DataFrame(np.array([model.feature_importances_]), columns=x.columns)

# ic(label)

label = label.transpose()

# ic(label)

label = label.sort_values(by=0)

label.columns = ['value']

# ic(label)

got_columns = label[:round(size * percentage)].index
# ic(got_columns)

x_train_tmp = x_train.copy()
x_test_tmp = x_test.copy()

# ic(x_train_tmp.shape, x_test_tmp.shape)

# ic(x_train_tmp.head(), x_test_tmp.tail())

x_train_tmp = x_train_tmp.drop(got_columns, axis=1)
x_test_tmp = x_test_tmp.drop(got_columns, axis=1)

# ic(x_train_tmp.shape, x_test_tmp.shape)
# ic(x_train_tmp.head(), x_test_tmp.tail())

print('(with less columns)')

model.fit(x_train_tmp, y_train)

ic(model.score(x_test_tmp, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test_tmp)
print('정답률 : ', accuracy_score(y_test, y_predict))
ic(model.feature_importances_)

'''
DecisionTreeClassifier
ic| model.score(x_test, y_test): 0.6244897959183674
정답률 :  0.6244897959183674
ic| model.feature_importances_: array([0.06887404, 0.1053567 , 0.08098317, 0.09600727, 0.07600551,
                                       0.10969526, 0.0950279 , 0.07495483, 0.09337104, 0.06749939,
                                       0.1322249 ])
(with less columns)
ic| model.score(x_test_tmp, y_test): 0.6193877551020408
정답률 :  0.6193877551020408
ic| model.feature_importances_: array([0.11501344, 0.09679143, 0.07653524, 0.08845413, 0.12762503,
                                       0.12357406, 0.09496401, 0.12295179, 0.15409086])

RandomForestClassifier
ic| model.score(x_test, y_test): 0.6989795918367347
정답률 :  0.6989795918367347
ic| model.feature_importances_: array([0.07556759, 0.09860754, 0.08047012, 0.08953323, 0.08400296,
                                       0.09419376, 0.09391752, 0.10687081, 0.08622927, 0.07792774,
                                       0.11267944])
(with less columns)
ic| model.score(x_test_tmp, y_test): 0.7
정답률 :  0.7
ic| model.feature_importances_: array([0.11395372, 0.09494574, 0.10804488, 0.10061011, 0.11227652,
                                       0.11115615, 0.12454602, 0.10541543, 0.12905141])
'''