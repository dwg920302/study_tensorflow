from icecream import ic
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor


dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

model = XGBClassifier()
# model = GradientBoostingClassifier()

model.fit(x_train, y_train)

ic(model.score(x_test, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
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
GradientBoosting
ic| model.score(x_test, y_test): 0.9298245614035088
정답률 :  0.9298245614035088
ic| model.feature_importances_: array([1.27262258e-06, 3.74738380e-02, 1.14847223e-03, 1.76893951e-04,
                                       1.31987348e-05, 2.35174419e-04, 1.28687939e-03, 1.77183769e-01,
                                       2.52751673e-03, 8.55339027e-06, 1.10929494e-02, 7.03750494e-05,
                                       6.11106153e-04, 1.52469241e-02, 3.24204242e-03, 2.17019270e-03,
                                       1.03176894e-05, 6.89660228e-04, 2.65803392e-05, 1.79500018e-03,
                                       5.74975486e-01, 2.81417234e-02, 1.10360010e-02, 3.12584890e-02,
                                       1.11743606e-02, 1.47090885e-03, 1.25209231e-02, 7.20256548e-02,
                                       6.73612743e-04, 1.71212391e-03])

XGB
ic| model.score(x_test, y_test): 0.9385964912280702
정답률 :  0.9385964912280702
ic| model.feature_importances_: array([8.7898206e-03, 2.7581811e-02, 0.0000000e+00, 1.9301679e-02,
                                       3.9901724e-03, 6.0011400e-03, 3.2148495e-02, 8.3173901e-02,
                                       1.1278887e-03, 4.0151473e-04, 4.1554882e-03, 2.1917797e-03,
                                       3.2115147e-02, 1.1539622e-02, 9.8985573e-04, 2.9151975e-03,
                                       2.9423193e-04, 1.0426682e-02, 2.5520625e-03, 6.3115481e-04,
                                       5.5882335e-01, 1.6619209e-02, 5.4739445e-02, 1.2755107e-02,
                                       1.6514983e-02, 7.0349168e-04, 1.4958775e-02, 6.8481632e-02,
                                       4.3315296e-03, 1.7447158e-03], dtype=float32)
'''