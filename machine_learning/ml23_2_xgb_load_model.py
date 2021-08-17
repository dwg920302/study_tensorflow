from icecream import ic
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt


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

model = XGBRegressor()
model.load_model('../_save/xgb_save/m23_xgb.dat')

score = model.score(x_test, y_test)
ic(score)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
ic(r2_score)

# history = model.evals_result()
# val_keys = history.keys()
# eval_keys = history['validation_0'].keys()

# for eval_set in eval_keys:
#     char = 'XGBoost with '+str(eval_set)
#     plt.title(char)
#     for val in val_keys:
#         data = np.array(history[val][eval_set])
#         plt.plot(range(data.shape[0]), data)
#     plt.show()
