from icecream import ic
import numpy as np

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

# Model

model = DecisionTreeClassifier(max_depth=8)
# model = RandomForestClassifier()

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
ic| model.score(x_test, y_test): 0.868421052631579
정답률 :  0.868421052631579
ic| model.feature_importances_: array([0.00623168, 0.06984559, 0.        , 0.        , 0.        ,
                                       0.0175266 , 0.0023012 , 0.        , 0.00305268, 0.        ,
                                       0.01841206, 0.00623168, 0.        , 0.01617038, 0.        ,
                                       0.        , 0.        , 0.01026395, 0.        , 0.        ,
                                       0.74702111, 0.01855758, 0.        , 0.        , 0.        ,
                                       0.        , 0.        , 0.08438548, 0.        , 0.        ])
'''

'''
RandomForestClassifier
ic| model.score(x_test, y_test): 0.9385964912280702
정답률 :  0.9385964912280702
ic| model.feature_importances_: array([0.065836  , 0.01205102, 0.04774314, 0.0614054 , 0.00683364,
                                       0.01547975, 0.04790564, 0.11547395, 0.00247247, 0.00454811,
                                       0.00637805, 0.0043056 , 0.01684525, 0.04427864, 0.00341464,
                                       0.00385782, 0.00732833, 0.00372885, 0.00235627, 0.00270645,
                                       0.11744697, 0.01648104, 0.12286616, 0.1172718 , 0.01334478,
                                       0.00545018, 0.02814746, 0.08568759, 0.00930334, 0.00905166])
'''