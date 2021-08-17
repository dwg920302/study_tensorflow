# coefficient 계수

import pandas as pd
from sklearn.linear_model import LinearRegression
from icecream import ic


x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

df = pd.DataFrame({'X':x, 'Y':y})
ic(df, df.shape)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
ic(x_train.shape, y_train.shape)
x_train = x_train.values.reshape(len(x_train), 1)
# y_train = y_train.values.reshape(len(y_train), 1)
ic(x_train.shape, y_train.shape)

model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
ic(score)

ic(model.coef_, model.intercept_) # 기울기 (weight), 절편 (bias)