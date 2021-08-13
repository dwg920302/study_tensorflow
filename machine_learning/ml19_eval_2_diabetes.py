from icecream import ic
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, PowerTransformer

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

dataset = load_diabetes()
x = dataset['data']
y = dataset['target']

ic(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=3/4, shuffle=True, random_state=16)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model = XGBRegressor()
model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=2)

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mape', 'rmsle'], eval_set=[(x_train, y_train), (x_test, y_test)])
# model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae', 'logloss'], eval_set=[(x_train, y_train), (x_test, y_test)]) # 뒤의 세트는 validation

'''
[0]     validation_0-rmse:21.58395      validation_1-rmse:21.70951
[1]     validation_0-rmse:19.54084      validation_1-rmse:19.62640
[2]     validation_0-rmse:17.70332      validation_1-rmse:17.76797
[3]     validation_0-rmse:16.04359      validation_1-rmse:16.06137
[4]     validation_0-rmse:14.55302      validation_1-rmse:14.57667
[5]     validation_0-rmse:13.20444      validation_1-rmse:13.20350
[6]     validation_0-rmse:11.99669      validation_1-rmse:11.98148
[7]     validation_0-rmse:10.90187      validation_1-rmse:10.88153
[8]     validation_0-rmse:9.92034       validation_1-rmse:9.95382
[9]     validation_0-rmse:9.03230       validation_1-rmse:9.07406
[10]    validation_0-rmse:8.23515       validation_1-rmse:8.29158

[246]   validation_0-rmse:0.06178       validation_1-rmse:2.68620
[247]   validation_0-rmse:0.06082       validation_1-rmse:2.68633
[248]   validation_0-rmse:0.05939       validation_1-rmse:2.68662
[249]   validation_0-rmse:0.05831       validation_1-rmse:2.68621
[250]   validation_0-rmse:0.05768       validation_1-rmse:2.68599
[251]   validation_0-rmse:0.05658       validation_1-rmse:2.68592
[252]   validation_0-rmse:0.05584       validation_1-rmse:2.68555
[253]   validation_0-rmse:0.05509       validation_1-rmse:2.68545
[254]   validation_0-rmse:0.05420       validation_1-rmse:2.68516
[255]   validation_0-rmse:0.05329       validation_1-rmse:2.68516

#XGBOOSTER의 기본 평가값은 rmse
'''

score = model.score(x_test, y_test)
ic(score)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
ic(r2_score)