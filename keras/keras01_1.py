# keras-001 [Keras 튜토리얼]

import numpy as np  # 설명 생략

from tensorflow.keras.models import Sequential  # 순차적 모델(Sequential)을 import함
from tensorflow.keras.layers import Dense   # Layer의 한 종류인 Dense를 import함. 기본 형태라고 봐도 됨


# 다음 순서대로 진행됨 (숙지해놓는 게 좋음)

# [Step 1] 데이터 준비 (전처리 과정 포함)
x = np.array([1,2,3])
y = np.array([1,2,3])

# [Step 2] 모델 정의
model = Sequential()    # 비어있는 Sequential 모델 정의
model.add(Dense(1, input_dim=1))    # 빈 Sequential 모델에 Dense 하나 추가

# [Step 3] 컴파일 및 훈련(fit)
model.compile(loss='mse', optimizer='adam') # 모델을 컴파일함.

model.fit(x, y, epochs=500, batch_size=1)  # 표본을 가지고 모델을 학습시킴.
# epochs=학습 회수, batch_size=학습시킬 표본의 단위 크기

# [Step 4] 평가(evaluate) or 예측(predict)
loss = model.evaluate(x, y) # 표본을 가지고 학습시킨 모델을 평가함.
print('loss : ', loss)

result = model.predict([4]) # 학습시킨 모델로 특정 표본(4)의 값을 예측함.
print('4의 예측값 : ', result)

'''
[Best Fit]
epochs=2500, batch_size=1
loss :  0.0
예측값 :  [[4.]]

[Epoch 500]
Epoch 500/500
3/3 [==============================] - 0s 499us/step - loss: 0.0364
1/1 [==============================] - 0s 67ms/step - loss: 0.0324
loss :  0.032430555671453476
4의 예측값 :  [[3.6039937]]
'''