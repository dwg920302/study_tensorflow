from icecream import ic
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # Tree의 확장형


dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality'].to_numpy()
x = dataset.drop(columns='quality')

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=37)

# Model

# model = DecisionTreeClassifier(max_depth=8)
model = RandomForestClassifier()

model.fit(x_train, y_train)

ic(model.score(x_test, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
ic(model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importance_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importance_dataset(model)
plt.show()

'''
ic| model.score(x_test, y_test): 0.5520408163265306
정답률 :  0.5520408163265306
ic| model.feature_importances_: array([0.05638359, 0.14335478, 0.06887511, 0.07176905, 0.07163709,
                                       0.09513188, 0.04945434, 0.03986439, 0.07530432, 0.06247046,
                                       0.26575499])
'''

'''
RandomForestClassifier
ic| model.score(x_test, y_test): 0.6989795918367347
정답률 :  0.6989795918367347
ic| model.feature_importances_: array([0.07536134, 0.09693064, 0.08309257, 0.0878075 , 0.08620539,
                                       0.09493185, 0.09199337, 0.10525229, 0.08628987, 0.07709727,
                                       0.11503791])
'''