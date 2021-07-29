# keras-007 [boston dataset]

# sklearn에 제공되는 Dataset 몇 개 중 하나

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston


# 데이터

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

# train test 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=47)

# 모델 구성

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(13))
model.add(Dense(1))


# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=500, verbose=1)
# batch_size (default 32)

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_pred = model.predict(x_test)
print('예측값 = ', y_pred)

# R2 구하기

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

'''
[res] (epochs=250, batch_size=1)

loss =  21.668197631835938
예측값 =
[[19.64466  ]
 [30.553722 ]
 [22.458921 ]
 [30.180878 ]
 [25.598576 ]
 [21.509043 ]
 [23.985958 ]
 [29.41997  ]
 [30.12049  ]
 [22.030336 ]
 [25.27116  ]
 [26.863047 ]
 [19.628664 ]
 [27.006458 ]
 [29.920887 ]
 [10.606451 ]
 [32.164463 ]
 [15.587378 ]
 [11.964382 ]
 [17.476486 ]
 [31.60826  ]
 [21.191366 ]
 [36.430573 ]
 [20.194893 ]
 [26.375992 ]
 [17.614029 ]
 [22.238434 ]
 [21.418188 ]
 [21.712132 ]
 [24.72517  ]
 [15.867928 ]
 [22.530262 ]
 [17.965584 ]
 [ 9.258223 ]
 [30.784239 ]
 [21.464788 ]
 [14.136105 ]
 [26.220938 ]
 [21.337551 ]
 [21.155193 ]
 [24.16099  ]
 [25.183699 ]
 [34.11074  ]
 [21.185785 ]
 [30.538387 ]
 [23.158392 ]
 [10.378284 ]
 [26.220613 ]
 [34.41252  ]
 [23.228016 ]
 [26.118477 ]
 [22.77206  ]
 [ 8.650545 ]
 [24.063255 ]
 [ 6.6565557]
 [23.77356  ]
 [18.416351 ]
 [15.297487 ]
 [ 7.6789765]
 [28.025366 ]
 [14.419428 ]
 [16.320105 ]
 [19.999084 ]
 [20.864836 ]
 [37.99946  ]
 [23.298384 ]
 [20.120678 ]
 [22.206236 ]
 [23.766918 ]
 [25.875048 ]
 [27.869268 ]
 [20.414375 ]
 [25.377369 ]
 [20.94527  ]
 [28.529932 ]
 [14.370142 ]
 [17.632708 ]
 [18.932144 ]
 [17.770267 ]
 [18.01508  ]
 [ 4.4915857]
 [33.017696 ]
 [30.294735 ]
 [23.998354 ]
 [29.980137 ]
 [16.152895 ]
 [23.12649  ]
 [27.463356 ]
 [25.191854 ]
 [21.186653 ]
 [18.897284 ]
 [19.942863 ]
 [15.250111 ]
 [16.814838 ]
 [28.750036 ]
 [20.083303 ]
 [37.287483 ]
 [28.466915 ]
 [ 6.9485545]
 [21.336975 ]
 [25.288845 ]
 [13.042898 ]
 [20.414745 ]
 [17.217339 ]
 [17.338083 ]
 [31.032654 ]
 [21.367086 ]
 [18.673395 ]
 [14.150232 ]
 [19.143658 ]
 [14.15419  ]
 [20.431507 ]
 [30.63725  ]
 [22.015404 ]
 [14.966171 ]
 [19.48162  ]
 [21.423979 ]
 [26.39201  ]
 [34.32758  ]
 [16.528843 ]
 [14.505474 ]
 [20.181086 ]
 [26.741152 ]
 [23.361485 ]
 [10.658577 ]
 [17.645407 ]
 [32.368008 ]]
R2 =  0.7303308278606828
'''