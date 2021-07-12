import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)

# (442, 10) (442,)

print(dataset.feature_names)

# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(dataset.DESCR)

print(np.min(y), np.max(y))

# 데이터 마저 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=47)

# 모델구성

model = Sequential()
model.add(Dense(128, input_shape=(10, )))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

# 컴파일 및 훔련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=8, epochs=200, verbose=2, validation_split=1/3, shuffle=True)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

'''
[엄청난 loss;;;]
batch_size=22, epochs=2200
loss :  3348.855224609375
예측값 =  [[134.15752 ]
 [121.09425 ]
 [ 79.1534  ]
 [161.24748 ]
 [ 87.563385]
 [244.30661 ]
 [157.1993  ]
 [279.8612  ]
 [146.32628 ]
 [121.01356 ]
 [152.05472 ]
 [221.49928 ]
 [ 83.06536 ]
 [304.1969  ]
 [ 84.612144]
 [148.32294 ]
 [ 83.12792 ]
 [151.23161 ]
 [164.71867 ]
 [ 87.814026]
 [118.12885 ]
 [219.02193 ]
 [114.55661 ]
 [195.36089 ]
 [231.36223 ]
 [225.21751 ]
 [ 95.95651 ]
 [192.08043 ]
 [252.33946 ]
 [ 58.88805 ]
 [193.20995 ]
 [133.1239  ]
 [209.05116 ]
 [165.80376 ]
 [192.77089 ]
 [169.47275 ]
 [180.4423  ]
 [227.2213  ]
 [157.86893 ]
 [212.59422 ]
 [ 94.19787 ]
 [149.6234  ]
 [188.7447  ]
 [189.81715 ]
 [116.21898 ]
 [207.86562 ]
 [237.06664 ]
 [175.33954 ]
 [121.88191 ]
 [ 52.055984]
 [141.97832 ]
 [116.65083 ]
 [153.43022 ]
 [ 67.75227 ]
 [176.58173 ]
 [204.09286 ]
 [125.54165 ]
 [148.39326 ]
 [191.5185  ]
 [245.10526 ]
 [143.79243 ]
 [ 73.05219 ]
 [232.51698 ]
 [205.74121 ]
 [161.49895 ]
 [221.96455 ]
 [161.67928 ]
 [ 99.24968 ]
 [201.33742 ]
 [125.7577  ]
 [126.16821 ]
 [102.842316]
 [240.757   ]
 [ 94.85003 ]
 [100.6074  ]
 [129.5617  ]
 [152.31487 ]
 [254.40123 ]
 [143.53328 ]
 [161.6836  ]
 [116.701485]
 [158.87503 ]
 [135.01399 ]
 [104.09245 ]
 [111.73774 ]
 [169.102   ]
 [167.82613 ]
 [199.8942  ]
 [172.8169  ]]
R2 =  0.41461809589939225
'''