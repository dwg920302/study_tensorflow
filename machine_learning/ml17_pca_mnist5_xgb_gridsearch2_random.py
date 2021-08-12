from icecream import ic
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score

from xgboost import XGBClassifier, XGBRegressor


# (x_train, _), (x_test, _) = mnist.load_data()
# _ it's called underscore and used for ignoring the specific values and others

(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, x_test.shape)

# x = np.append(x_train, x_test, axis=0)
# y = np.append(y_train, y_test, axis=0)

# ic(x.shape)

# x_shape = x.shape

x_shape = x_train.shape

# 실습
# xgb를 통해, 0.999 이상이 되는 n_component를 사용하여
# 기존보다 성능 향상, RandomSearch로도 해볼 것

pca = PCA()

x_train = pca.fit_transform(x_train.reshape(x_shape[0], x_shape[1]*x_shape[2]))
x_test = pca.transform(x_test.reshape(x_test.shape[0], x_shape[1]*x_shape[2]))

pca_evr = pca.explained_variance_ratio_
ic(pca_evr, sum(pca_evr))

cumsum = np.cumsum(pca_evr)
ic(cumsum) 

ic(np.argmax(cumsum >= 0.999)+1)

n_components = np.argmax(cumsum >= 0.999)+1

pca = PCA(n_components=n_components)  # 154

# 카테고리컬 안 시킴

x_train_2 = pca.fit_transform(x_train)
x_test_2 = pca.transform(x_test)

scaler = MinMaxScaler()
x_train_2 = scaler.fit_transform(x_train_2)
x_test_2 = scaler.transform(x_test_2)

parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.01, 0.001], "max_depth":[4, 5, 6]},
    {"n_estimators":[80, 120, 160], "learning_rate":[0.1, 0.01, 0.001], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.8, 1]},
    {"n_estimators":[80, 120, 160], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.8, 1], "colsample_bylevel":[0.6, 0.8, 1]}
]

kfold = KFold(n_splits=5, shuffle=True, random_state=37)

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1, n_iter=25)
# model = XGBRegressor() # 안댐

model.fit(x_train_2, y_train)

ic(model.score(x_test_2, y_test)) # ic| model.score(x_test, y_test): 0.9666666666666667 # evaluate

y_predict = model.predict(x_test_2)
print('acc : ', accuracy_score(y_test, y_predict))
print('r2 : ', r2_score(y_test, y_predict))
ic(model.feature_importances_)
'''
ic| model.score(x_test_2, y_test): 0.9603
acc :  0.9603
r2 :  0.9154223750063988
'''