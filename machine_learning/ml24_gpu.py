from icecream import ic
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from time import time


dataset = load_boston()
x = dataset['data']
y = dataset['target']

ic(x.shape, y.shape)
# ic| x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model = XGBRegressor()
# model = XGBRegressor(
#     n_estimators=256,
#     learning_rate=0.1,
#     n_jobs=2)

model = XGBRegressor(
    n_estimators=256,
    learning_rate=0.1,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0)

start_time = time()

model.fit(x_train, y_train, verbose=1, eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=10)   # 간단하게 숫자만 적어주면 됨. rounds = patience
# model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae', 'logloss'], eval_set=[(x_train, y_train), (x_test, y_test)]) # 뒤의 세트는 validation

elapsed_time = time() - start_time

ic(elapsed_time)

# ic| elapsed_time: 0.19248461723327637
# 학원 / 집컴과 비교해보기

score = model.score(x_test, y_test)
ic(score)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
ic(r2_score)

history = model.evals_result()
val_keys = history.keys()
eval_keys = history['validation_0'].keys()

for eval_set in eval_keys:
    char = 'XGBoost with '+str(eval_set)
    plt.title(char)
    for val in val_keys:
        data = np.array(history[val][eval_set])
        plt.plot(range(data.shape[0]), data)
    plt.show()