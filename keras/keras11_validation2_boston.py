# keras-011 #2 [Boston with validation_split]

from icecream import ic

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import r2_score


# 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

ic(x.shape, y.shape)

ic(datasets.feature_names)
ic(datasets.DESCR)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# :Attribute Information (in order):
#     - CRIM     per capita crime rate by town
#     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#     - INDUS    proportion of non-retail business acres per town
#     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#     - NOX      nitric oxides concentration (parts per 10 million)
#     - RM       average number of rooms per dwelling
#     - AGE      proportion of owner-occupied units built prior to 1940
#     - DIS      weighted distances to five Boston employment centres
#     - RAD      index of accessibility to radial highways
#     - TAX      full-value property-tax rate per $10,000
#     - PTRATIO  pupil-teacher ratio by town
#     - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#     - LSTAT    % lower status of the population
#     - MEDV     Median value of owner-occupied homes in $1000's

# train test 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True)

# 모델

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(13))
model.add(Dense(1))

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=13, epochs=600, validation_split=1/6, shuffle=True, verbose=0)
# batch_size (default 32)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
ic(loss)

y_pred = model.predict(x_test)
print('예측값 = ', y_pred)

# R2 구하기

r2_score = r2_score(y_test, y_pred)
ic(r2_score)

'''
[res] (epochs=250, batch_size=1)

R2 =  0.7303308278606828
'''

#validation 좋은 지 모르겠음 -> validation 쓸만하네 ㅋㅋㅋ