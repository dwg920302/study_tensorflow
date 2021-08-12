# 실습
# Feature Importance(줄여서 FI)가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋을 재구성 후
# 각 모델별로 결과 도출, 기존 모델과 비교
# DataFrame

from icecream import ic
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # Tree의 확장형


dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

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
# ic(got_columns)

x_train_tmp = x_train.copy()
x_test_tmp = x_test.copy()

# ic(x_train_tmp.shape, x_test_tmp.shape)

# ic(x_train_tmp.head(), x_test_tmp.tail())

x_train_tmp = x_train_tmp.drop(got_columns, axis=1)
x_test_tmp = x_test_tmp.drop(got_columns, axis=1)

# ic(x_train_tmp.shape, x_test_tmp.shape)
# ic(x_train_tmp.head(), x_test_tmp.tail())

model.fit(x_train_tmp, y_train)

ic(model.score(x_test_tmp, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test_tmp)
print('정답률 : ', accuracy_score(y_test, y_predict))
ic(model.feature_importances_)

'''
DecisionTreeClassifier
ic| model.score(x_test, y_test): 0.8771929824561403
정답률 :  0.8771929824561403
ic| model.feature_importances_: array([0.        , 0.05211279, 0.        , 0.        , 0.        ,
                                       0.        , 0.01982781, 0.        , 0.00305268, 0.        ,
                                       0.01841206, 0.        , 0.        , 0.00915974, 0.        ,
                                       0.        , 0.01246336, 0.01026395, 0.        , 0.        ,
                                       0.74702111, 0.04330102, 0.        , 0.        , 0.        ,
(with less columns)
ic| model.score(x_test_tmp, y_test): 0.8596491228070176
정답률 :  0.8596491228070176
ic| model.feature_importances_: array([0.04510215, 0.00701064, 0.        , 0.        , 0.        ,
                                       0.0023012 , 0.0175266 , 0.00305268, 0.00623168, 0.01841206,
                                       0.        , 0.        , 0.00915974, 0.        , 0.        ,
                                       0.        , 0.01649563, 0.        , 0.        , 0.74702111,
                                       0.04330102, 0.08438548, 0.        , 0.        ])

RandomForestClassifier
  model.fit(x_train, y_train)
ic| model.score(x_test, y_test): 0.9473684210526315
정답률 :  0.9473684210526315
ic| model.feature_importances_: array([0.04942779, 0.01376133, 0.04164142, 0.08899598, 0.0048212 ,
                                       0.00648215, 0.03544951, 0.11166956, 0.00453298, 0.00242211,
                                       0.01149787, 0.00281019, 0.00624267, 0.07507131, 0.00389489,
                                       0.00334099, 0.00497136, 0.01178661, 0.00379033, 0.00568529,
                                       0.11285543, 0.01489511, 0.10278982, 0.12712967, 0.0111258 ,
                                       0.01040272, 0.02935801, 0.0883647 , 0.00962571, 0.00515748]
(with less columns)
ic| model.score(x_test_tmp, y_test): 0.9298245614035088
정답률 :  0.9298245614035088
ic| model.feature_importances_: array([0.05398246, 0.01372305, 0.04693087, 0.0603762 , 0.00899887,
                                       0.01380781, 0.02139711, 0.14910034, 0.01076968, 0.02563717,
                                       0.04340472, 0.00522865, 0.00480149, 0.00502062, 0.11735929,
                                       0.01520956, 0.13963195, 0.12249166, 0.01183076, 0.00804024,
                                       0.04295798, 0.06029983, 0.01347495, 0.00552474])
'''