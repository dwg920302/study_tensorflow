from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)
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

# 데이터는 주어졌으니 모델

# model = Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(13))
# model.add(Dense(1))

input_1 = Input(shape=(13, ))
dense_1 = Dense(8)(input_1)
dense_2 = Dense(16)(dense_1)
dense_3 = Dense(4)(dense_2)
output_1 = Dense(1)(dense_3)
model = Model(inputs = input_1, outputs = output_1)

# model.summary()

# train test 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=47)

# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=500, verbose=0)
# batch_size (default 32)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_pred = model.predict(x_test)
print('예측값 = ', y_pred)

# R2 구하기

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)