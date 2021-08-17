# 실습
# 1. 상단 모델에 GridSearch 또는 RandomSearch로 튜닝한 Model 구성
# 최적의 R2값과 FI를 구할 것

# 2. 위 스레드값으로 SelectFromModel 돌려서 최적의 Feature 갯수 구하기

# 3. 1번값과 4번값 비교, r2 0.47 이상

from icecream import ic
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRegressor

from sklearn.metrics import r2_score


# Data
x, y = load_boston(return_X_y=True)

ic(x.shape, y.shape)
# ic| x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

# Model

parameters = [

    {"gamma":[0.01, 0.001, 0.0001],
    "n_estimators":[100, 200, 300],
    "learning_rate":[0.1, 0.01, 0.001],
    "max_depth":[4, 5, 6]}
]

# RamdomizedSearchCV로 감싸기

kfold = KFold(n_splits=5, shuffle=True, random_state=27)

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1)

# model = GridSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1)

# Fit

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
ic(score)

# ic(model.best_estimator_)

best_model = model.best_estimator_

best_model.fit(x_train, y_train)

# here Error
thresholds = np.sort(best_model.feature_importances_)
ic(thresholds)

for thresh in thresholds:
    ic(thresh)
    selection = SelectFromModel(best_model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    ic(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))


'''
[RandomSearch]
ic| score: 0.9190049118975776
ic| thresholds: array([0.00252118, 0.00831071, 0.01052447, 0.0111452 , 0.01276798,
                       0.01390982, 0.02003548, 0.02512315, 0.03878516, 0.03894828,
                       0.04200922, 0.31237662, 0.4635427 ], dtype=float32)
ic| thresh: 0.0025211798
ic| select_x_train.shape: (404, 13), select_x_test.shape: (102, 13)
Thresh=0.003, n=13, R2: 89.55%
ic| thresh: 0.008310705
ic| select_x_train.shape: (404, 12), select_x_test.shape: (102, 12)
Thresh=0.008, n=12, R2: 89.55%
ic| thresh: 0.010524474
ic| select_x_train.shape: (404, 11), select_x_test.shape: (102, 11)
Thresh=0.011, n=11, R2: 91.35%
ic| thresh: 0.011145196
ic| select_x_train.shape: (404, 10), select_x_test.shape: (102, 10)
Thresh=0.011, n=10, R2: 91.25%
ic| thresh: 0.012767977
ic| select_x_train.shape: (404, 9), select_x_test.shape: (102, 9)
Thresh=0.013, n=9, R2: 90.68%
ic| thresh: 0.01390982
ic| select_x_train.shape: (404, 8), select_x_test.shape: (102, 8)
Thresh=0.014, n=8, R2: 89.01%
ic| thresh: 0.020035475
ic| select_x_train.shape: (404, 7), select_x_test.shape: (102, 7)
Thresh=0.020, n=7, R2: 89.46%
ic| thresh: 0.025123153
ic| select_x_train.shape: (404, 6), select_x_test.shape: (102, 6)
Thresh=0.025, n=6, R2: 87.88%
ic| thresh: 0.038785156
ic| select_x_train.shape: (404, 5), select_x_test.shape: (102, 5)
Thresh=0.039, n=5, R2: 88.56%
ic| thresh: 0.03894828
ic| select_x_train.shape: (404, 4), select_x_test.shape: (102, 4)
Thresh=0.039, n=4, R2: 87.24%
ic| thresh: 0.042009216
ic| select_x_train.shape: (404, 3), select_x_test.shape: (102, 3)
Thresh=0.042, n=3, R2: 81.72%
ic| thresh: 0.31237662
ic| select_x_train.shape: (404, 2), select_x_test.shape: (102, 2)
Thresh=0.312, n=2, R2: 68.71%
ic| thresh: 0.4635427
ic| select_x_train.shape: (404, 1), select_x_test.shape: (102, 1)
Thresh=0.464, n=1, R2: 49.24%


[GridSearch]

ic| score: 0.9222652446588573
ic| thresholds: array([0.00357572, 0.01020357, 0.01038477, 0.01224723, 0.01298863,
                       0.02083744, 0.02759543, 0.03164344, 0.03929441, 0.04610997,
                       0.05114788, 0.2989721 , 0.4349994 ], dtype=float32)
ic| thresh: 0.0035757213
ic| select_x_train.shape: (404, 13), select_x_test.shape: (102, 13)
Thresh=0.004, n=13, R2: 89.55%
ic| thresh: 0.010203567
ic| select_x_train.shape: (404, 12), select_x_test.shape: (102, 12)
Thresh=0.010, n=12, R2: 89.55%
ic| thresh: 0.010384766
ic| select_x_train.shape: (404, 11), select_x_test.shape: (102, 11)
Thresh=0.010, n=11, R2: 90.30%
ic| thresh: 0.012247226
ic| select_x_train.shape: (404, 10), select_x_test.shape: (102, 10)
Thresh=0.012, n=10, R2: 90.14%
ic| thresh: 0.012988629
ic| select_x_train.shape: (404, 9), select_x_test.shape: (102, 9)
Thresh=0.013, n=9, R2: 89.77%
ic| thresh: 0.020837437
ic| select_x_train.shape: (404, 8), select_x_test.shape: (102, 8)
Thresh=0.021, n=8, R2: 89.01%
ic| thresh: 0.027595429
ic| select_x_train.shape: (404, 7), select_x_test.shape: (102, 7)
Thresh=0.028, n=7, R2: 88.14%
ic| thresh: 0.031643443
ic| select_x_train.shape: (404, 6), select_x_test.shape: (102, 6)
Thresh=0.032, n=6, R2: 87.88%
ic| thresh: 0.03929441
ic| select_x_train.shape: (404, 5), select_x_test.shape: (102, 5)
Thresh=0.039, n=5, R2: 88.56%
ic| thresh: 0.046109974
ic| select_x_train.shape: (404, 4), select_x_test.shape: (102, 4)
Thresh=0.046, n=4, R2: 85.55%
ic| thresh: 0.05114788
ic| select_x_train.shape: (404, 3), select_x_test.shape: (102, 3)
Thresh=0.051, n=3, R2: 81.98%
ic| thresh: 0.2989721
ic| select_x_train.shape: (404, 2), select_x_test.shape: (102, 2)
Thresh=0.299, n=2, R2: 68.71%
ic| thresh: 0.4349994
ic| select_x_train.shape: (404, 1), select_x_test.shape: (102, 1)
Thresh=0.435, n=1, R2: 49.24%
'''