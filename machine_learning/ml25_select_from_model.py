from icecream import ic
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

from time import time

from sklearn.feature_selection import SelectFromModel


# dataset = load_boston()
# x = dataset['data']
# y = dataset['target']
x, y = load_boston(return_X_y=True)

ic(x.shape, y.shape)
# ic| x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

model = XGBRegressor()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
ic(score)

thresholds = np.sort(model.feature_importances_)
ic(thresholds)

for thresh in thresholds:
    ic(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    ic(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))
'''
ic| thresh: 0.0013380182
ic| select_x_train.shape: (404, 13), select_x_test.shape: (102, 13)
Thresh=0.001, n=13, R2: 89.55%
ic| thresh: 0.002985412
ic| select_x_train.shape: (404, 12), select_x_test.shape: (102, 12)
Thresh=0.003, n=12, R2: 89.55%
ic| thresh: 0.008811081
ic| select_x_train.shape: (404, 11), select_x_test.shape: (102, 11)
Thresh=0.009, n=11, R2: 89.89%
ic| thresh: 0.010838771
ic| select_x_train.shape: (404, 10), select_x_test.shape: (102, 10)
Thresh=0.011, n=10, R2: 90.21%
ic| thresh: 0.011978533
ic| select_x_train.shape: (404, 9), select_x_test.shape: (102, 9)
Thresh=0.012, n=9, R2: 89.77%
ic| thresh: 0.015193855
ic| select_x_train.shape: (404, 8), select_x_test.shape: (102, 8)
Thresh=0.015, n=8, R2: 90.19%
ic| thresh: 0.019045196
ic| select_x_train.shape: (404, 7), select_x_test.shape: (102, 7)
Thresh=0.019, n=7, R2: 87.01%
ic| thresh: 0.021851087
ic| select_x_train.shape: (404, 6), select_x_test.shape: (102, 6)
Thresh=0.022, n=6, R2: 87.88%
ic| thresh: 0.03366318
ic| select_x_train.shape: (404, 5), select_x_test.shape: (102, 5)
Thresh=0.034, n=5, R2: 86.88%
ic| thresh: 0.06317842
ic| select_x_train.shape: (404, 4), select_x_test.shape: (102, 4)
Thresh=0.063, n=4, R2: 85.55%
ic| thresh: 0.06863533
ic| select_x_train.shape: (404, 3), select_x_test.shape: (102, 3)
Thresh=0.069, n=3, R2: 81.72%
ic| thresh: 0.26415828
ic| select_x_train.shape: (404, 2), select_x_test.shape: (102, 2)
Thresh=0.264, n=2, R2: 68.71%
ic| thresh: 0.4783228
ic| select_x_train.shape: (404, 1), select_x_test.shape: (102, 1)
Thresh=0.478, n=1, R2: 49.24%
'''