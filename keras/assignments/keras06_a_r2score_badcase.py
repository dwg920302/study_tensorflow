# keras-006 #1-2 [r2 score]

'''
1 R2를 음수가 아닌 0.5 이하로 만들기 (나쁘게)
2 데이터 건드리지 않기
3 레이어는 Input output 포함 6개 이상
4 배치 사이즈 = 1(exact) / # 5 epo = 100 이상
6 Hidden Layer의 Node는 10개 이상 1000개 이하
7 train 비율 70%
'''

from icecream import ic

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.metrics import r2_score


# 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

print('x = ', x_train, ', ' ,x_test)
print('y = ', y_train, ', ' ,y_test)

print('x = ', x_train.shape, ', ' ,x_test.shape)
print('y = ', y_train.shape, ', ' ,y_test.shape)

# 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(25))
model.add(Dense(1000))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dropout(0.8))
model.add(Dropout(0.5))
# Dropout -> 
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 평가, 예측
# loss = model.evaluate(x_test, y_test)
loss = model.evaluate(x_test, y_test)
ic(loss)

# result = model.predict([11])
# print('예측값 : ', result)

y_predict = model.predict(x_test)
print('예측값 = ' ,y_predict)

r2_score = r2_score(y_test, y_predict)
ic(r2_score)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

'''
[Best Worst Fit]
epochs=100, batch_size=1
loss :  611.025146484375
예측값 =  [[35.14232  ]
 [ 7.6467104]
 [44.30752  ]
 [52.45437  ]
 [ 8.155888 ]
 [23.940405 ]
 [19.866978 ]
 [10.1926   ]
 [41.76163  ]
 [32.596428 ]
 [15.793556 ]
 [13.247666 ]
 [31.57807  ]
 [12.229312 ]
 [28.523006 ]
 [26.99547  ]
 [24.44958  ]
 [38.706562 ]
 [24.958757 ]
 [47.871773 ]
 [ 3.0641088]
 [34.633137 ]
 [44.816696 ]
 [33.61478  ]
 [42.270805 ]
 [19.357801 ]
 [21.394512 ]
 [45.83506  ]
 [ 4.5916424]
 [ 5.6099987]]
r2 =  0.22062671407706014
'''