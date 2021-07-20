import numpy as np
from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping

# Data

x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11],
            [10,11,12], [20,30,40], [30,40,50], [40,50,60]])

x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80],
            [70,80,90], [80,90,100], [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([50, 60, 70])
x2_predict = np.array([65, 75, 85])



input_1 = Input(shape=(3, 1))
dense_1_1 = LSTM(8, activation='relu')(input_1)
output_1 = Dense(8, name='denseA4')(dense_1_1)

input_2 = Input(shape=(3, 1))
dense_2_1 = LSTM(8, activation='relu')(input_2)
output_2 = Dense(8)(dense_2_1)

merge_1 = concatenate([output_1, output_2])
merge_2 = Dense(16)(merge_1)

# 분기점

merge_3_1 = Dense(8)(merge_2)
last_output_1 = Dense(1, name='output-1')(merge_3_1)

merge_3_2 = Dense(4)(merge_2)
last_output_2 = Dense(1, name='output-2')(merge_3_2)

model = Model(inputs=[input_1, input_2], outputs=[last_output_1, last_output_2])

model.summary()

# 컴파일

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit([x1, x2], y, epochs=100, batch_size=1)

# Predict

res = model.predict([x1_predict, x2_predict])

print(res)

'''
[array([[35.469227],
       [43.61012 ],
       [51.926704]], dtype=float32), array([[36.003048],
       [44.33186 ],
       [52.84001 ]], dtype=float32)]
'''