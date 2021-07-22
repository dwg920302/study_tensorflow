import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array([range(1001, 1101)])
y1 = np.transpose(y1)

print(x1.shape, x2.shape, y1.shape)

# 모델

input_1 = Input(shape=(3, ))
dense_1_1 = Dense(10, activation='relu', name='denseA1')(input_1)
dense_1_2 = Dense(7, activation='relu', name='denseA2')(dense_1_1)
dense_1_3 = Dense(5, activation='relu', name='denseA3')(dense_1_2)
output_1 = Dense(11, name='denseA4')(dense_1_3)

input_2 = Input(shape=(3, ))
dense_2_1 = Dense(10, activation='relu')(input_2)
dense_2_2 = Dense(10, activation='relu')(dense_2_1)
dense_2_3 = Dense(10, activation='relu')(dense_2_2)
dense_2_4 = Dense(10, activation='relu')(dense_2_3)
output_2 = Dense(12)(dense_2_4)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.85, shuffle=True, random_state=24)

# 정의 순서는 Train, Test, Train, Test, ...

merge_1 = concatenate([output_1, output_2])
merge_2 = Dense(2, name='dense_beforelast')(merge_1)
last_output = Dense(1)(merge_2)
model = Model(inputs=[input_1, input_2], outputs=last_output)

# output은 2개의 output을 concatenate함

model.summary()

# 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, batch_size=10, epochs=100, verbose=1, validation_split=3/17, shuffle=True)

# 평가, 예측
results = model.evaluate([x1_test, x2_test], y1_test)
print('loss : ', results[0])
print('mae : ', results[1])

y_predict = model.predict([x1, x2])

r2 = r2_score(y1, y_predict)
print(r2)


'''
[Best Fit]
batch_size=10, epochs=1000
loss :  1.1175870895385742e-08
mae :  8.544921729480848e-05
0.9999999999852464
'''

'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           40          input_2[0][0]
__________________________________________________________________________________________________
denseA1 (Dense)                 (None, 10)           40          input_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           110         dense[0][0]
__________________________________________________________________________________________________
denseA2 (Dense)                 (None, 7)            77          denseA1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           110         dense_1[0][0]
__________________________________________________________________________________________________
denseA3 (Dense)                 (None, 5)            40          denseA2[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 10)           110         dense_2[0][0]
__________________________________________________________________________________________________
denseA4 (Dense)                 (None, 11)           66          denseA3[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 12)           132         dense_3[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 23)           0           denseA4[0][0]
                                                                 dense_4[0][0]
__________________________________________________________________________________________________
dense_beforelast (Dense)        (None, 2)            48          concatenate[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            3           dense_beforelast[0][0]
==================================================================================================
Total params: 776
Trainable params: 776
Non-trainable params: 0
__________________________________________________________________________________________________
'''