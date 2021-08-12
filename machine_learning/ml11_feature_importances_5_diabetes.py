from icecream import ic
import numpy as np

from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # Tree의 확장형


dataset = load_diabetes()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

# model = DecisionTreeRegressor(max_depth=2)
model = RandomForestRegressor()

model.fit(x_train, y_train)

ic(model.score(x_test, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test)
print('정답률 : ', r2_score(y_test, y_predict))
ic(model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importance_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importance_dataset(model)
plt.show()

'''
depth 2
ic| model.score(x_test, y_test): 0.43402621702956923
정답률 :  0.43402621702956923
ic| model.feature_importances_: array([0.        , 0.        , 0.35325336, 0.        , 0.        ,
                                       0.        , 0.        , 0.        , 0.64674664, 0.        ])

depth 4
ic| model.score(x_test, y_test): 0.3373817336075433
정답률 :  0.3373817336075433
ic| model.feature_importances_: array([0.021094  , 0.        , 0.33864877, 0.04151782, 0.        ,
                                       0.05711929, 0.01255439, 0.        , 0.52906573, 0.        ])

depth 8
ic| model.score(x_test, y_test): 0.13332970533309296
정답률 :  0.13332970533309296
ic| model.feature_importances_: array([0.0576194 , 0.00733769, 0.28451414, 0.06634751, 0.05191045,
                                       0.0578483 , 0.04719938, 0.01209348, 0.3602572 , 0.05487245])
'''

'''
RandomForestClassifier
ic| model.score(x_test, y_test): 0.49536191504320015
정답률 :  0.49536191504320015
ic| model.feature_importances_: array([0.06320394, 0.01353457, 0.25321804, 0.12232411, 0.04626274,
                                       0.05420891, 0.06014212, 0.02646999, 0.29421203, 0.06642356])
'''