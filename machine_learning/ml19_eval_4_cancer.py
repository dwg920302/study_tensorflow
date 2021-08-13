# eval_metric을 찾아서 추가

from icecream import ic
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']

ic(x.shape, y.shape)
# ic| x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_estimators=1024, learning_rate=0.001, n_jobs=2)

# model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss', 'merror'], eval_set=[(x_train, y_train), (x_test, y_test)])
model.fit(x_train, y_train, verbose=1, eval_set=[(x_train, y_train), (x_test, y_test)])
# 이건 

'''
[11]    validation_0-logloss:0.68240    validation_1-logloss:0.68332
[12]    validation_0-logloss:0.68152    validation_1-logloss:0.68250
[13]    validation_0-logloss:0.68064    validation_1-logloss:0.68171
[14]    validation_0-logloss:0.67975    validation_1-logloss:0.68090
[15]    validation_0-logloss:0.67888    validation_1-logloss:0.68011
[16]    validation_0-logloss:0.67800    validation_1-logloss:0.67930
[17]    validation_0-logloss:0.67712    validation_1-logloss:0.67849
[18]    validation_0-logloss:0.67625    validation_1-logloss:0.67769
[19]    validation_0-logloss:0.67537    validation_1-logloss:0.67690
[20]    validation_0-logloss:0.67450    validation_1-logloss:0.67610

[1014]  validation_0-logloss:0.23658    validation_1-logloss:0.28221
[1015]  validation_0-logloss:0.23636    validation_1-logloss:0.28201
[1016]  validation_0-logloss:0.23614    validation_1-logloss:0.28182
[1017]  validation_0-logloss:0.23592    validation_1-logloss:0.28164
[1018]  validation_0-logloss:0.23571    validation_1-logloss:0.28146
[1019]  validation_0-logloss:0.23549    validation_1-logloss:0.28126
[1020]  validation_0-logloss:0.23528    validation_1-logloss:0.28107
[1021]  validation_0-logloss:0.23506    validation_1-logloss:0.28088
[1022]  validation_0-logloss:0.23485    validation_1-logloss:0.28069
[1023]  validation_0-logloss:0.23463    validation_1-logloss:0.28051

#이건 logloss가 default
'''

score = model.score(x_test, y_test)
ic(score)

y_predict = model.predict(x_test)
accuracy_score = accuracy_score(y_test, y_predict)
ic(accuracy_score)

# ic| score: 0.9649122807017544
# ic| accuracy_score: 0.9649122807017544