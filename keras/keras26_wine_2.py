import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt, font_manager, rc

# 한글깨짐 해결
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 다중분류 모델링하고, Acc 0.8 이상 완성

# 파이썬, 넘파이 ()
# x와 y 분리
# sklearn의 onehot 사용하기
# y의 라벨을 확인 (y.unique())

dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality']
x = dataset.drop(columns='quality')

print(x, y)
print(x.shape, y.shape)

print(y.unique()) # 7종류 [3,4,5,6,7,8,9]

# to_categorical을 그냥 쓰면 7개가 아니라 10개가 생성됨

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

# OneHotEncoder 이놈 매우 까탈스럽네 ㅡㅡ

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.755, shuffle=True, random_state=17)

# Model
input_1 = Input(shape=(11, ))
dense_1 = Dense(32, activation='relu')(input_1)
dense_2 = Dense(128)(dense_1)
dense_3 = Dense(512)(dense_2)
dense_4 = Dense(1024, activation='selu')(dense_3)
dense_5 = Dense(1024, activation='relu')(dense_4)
dense_6 = Dense(1024, activation='elu')(dense_5)
dense_7 = Dense(256)(dense_6)
dense_8 = Dense(64, activation='relu')(dense_7)
output_1 = Dense(7, activation='softmax')(dense_8)
model = Model(inputs = input_1, outputs = output_1)
    

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

es = EarlyStopping(monitor='accuracy', patience=100, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=32, epochs=1000, verbose=2, shuffle=True, validation_split=1/151, callbacks=[es])

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# y_pred = model.predict(x_test[-5:])
# print(y_pred, '\n', y_test[-5:])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])

plt.title('정확도 [Train, Validation]')
plt.xlabel('시행회수')
plt.ylabel('정확도(Train), 정확도(Validation)')
plt.legend(['정확도(Train)', '정확도(Validation)'])
plt.show()


'''
[Best Fit]
with StandardScaler, Tree 11 > 128 > 512 > 1024 > 1024 > 1024 > 256 > 64 > 7
batch_size=32, epochs=100
train : test : val = 6 : 2 : 2
loss =  2.0802271366119385 , accuracy =  0.6299319863319397

with MaxAbsScaler, Tree 11 > 128 > 512 > 1024 > 1024 > 1024 > 256 > 64 > 7
batch_size=32, epochs=100
loss =  2.213486909866333 , accuracy =  0.680272102355957
'''