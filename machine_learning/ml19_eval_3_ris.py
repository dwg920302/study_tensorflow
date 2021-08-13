# eval_metric을 찾아서 추가

from icecream import ic
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

dataset = load_iris()
x = dataset['data']
y = dataset['target']

ic(x.shape, y.shape)
# ic| x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_estimators=64, learning_rate=0.001, n_jobs=2)

model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss', 'merror'], eval_set=[(x_train, y_train), (x_test, y_test)])
# model.fit(x_train, y_train, verbose=1, eval_set=[(x_train, y_train), (x_test, y_test)])

# rmse류는 에러가 나고, default로 

'''
[0]     validation_0-mlogloss:1.09724   validation_1-mlogloss:1.09735
[1]     validation_0-mlogloss:1.09587   validation_1-mlogloss:1.09609
[2]     validation_0-mlogloss:1.09450   validation_1-mlogloss:1.09483
[3]     validation_0-mlogloss:1.09313   validation_1-mlogloss:1.09357
[4]     validation_0-mlogloss:1.09177   validation_1-mlogloss:1.09231
[5]     validation_0-mlogloss:1.09041   validation_1-mlogloss:1.09106
[6]     validation_0-mlogloss:1.08905   validation_1-mlogloss:1.08981
[7]     validation_0-mlogloss:1.08769   validation_1-mlogloss:1.08856
[8]     validation_0-mlogloss:1.08634   validation_1-mlogloss:1.08732
[9]     validation_0-mlogloss:1.08499   validation_1-mlogloss:1.08607

[54]    validation_0-mlogloss:1.02649   validation_1-mlogloss:1.03225
[55]    validation_0-mlogloss:1.02524   validation_1-mlogloss:1.03110
[56]    validation_0-mlogloss:1.02399   validation_1-mlogloss:1.02995
[57]    validation_0-mlogloss:1.02275   validation_1-mlogloss:1.02880
[59]    validation_0-mlogloss:1.02026   validation_1-mlogloss:1.02652
[60]    validation_0-mlogloss:1.01902   validation_1-mlogloss:1.02538
[61]    validation_0-mlogloss:1.01778   validation_1-mlogloss:1.02424
[62]    validation_0-mlogloss:1.01655   validation_1-mlogloss:1.02310
[63]    validation_0-mlogloss:1.01531   validation_1-mlogloss:1.02197

#XGBOOSTER의 기본 평가값은 rmse..가 아니었고, 맞는 걸 자동으로 하나 찾아줌.
#boston의 경우에는 rmse였지만 iris에서는 mlogloss
'''

score = model.score(x_test, y_test)
ic(score)

y_predict = model.predict(x_test)
accuracy_score = accuracy_score(y_test, y_predict)
ic(accuracy_score)