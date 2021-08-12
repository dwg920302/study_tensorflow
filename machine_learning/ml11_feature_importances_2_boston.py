from icecream import ic
import numpy as np

from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # Tree의 확장형


dataset = load_boston()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

model = DecisionTreeRegressor(max_depth=8)
# model = RandomForestRegressor()

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
0.8 / 0.2
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.01335281, 0.01669101, 0.89606565, 0.07389054])
# 각 컬럼에 대한 feature importances

0.7 / 0.3
ic| model.score(x_test, y_test): 0.9777777777777777
정답률 :  0.9777777777777777
ic| model.feature_importances_: array([0.        , 0.03818876, 0.37997483, 0.58183641])

max depth 4
ic| model.score(x_test, y_test): 1.0
정답률 :  1.0
ic| model.feature_importances_: array([0.        , 0.017208  , 0.40059131, 0.58220069])

max depth 8
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.01669101, 0.02956693, 0.89654253, 0.05719953])
'''

'''
RandomForestClassifier
ic| model.score(x_test, y_test): 0.9666666666666667
정답률 :  0.9666666666666667
ic| model.feature_importances_: array([0.08382746, 0.02886205, 0.46046303, 0.42684746])
'''