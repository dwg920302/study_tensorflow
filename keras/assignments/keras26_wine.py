import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler

# HW 2
# 수단과 방법을 가리지 않고 Accuracy 0.8 이상 만들기

dataset = load_wine()

print(dataset.DESCR)
print(dataset.feature_names)

# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (178, 13) (178,)

# Class가 3개이므로 다중 분석을 해야 함. 그에 따라 데이터를 카테고리화함. (후에 컴파일 시 categorical_crossent.를 사용)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=36)

# Model
input_1 = Input(shape=(13, ))
dense_1 = Dense(64)(input_1)
dense_2 = Dense(256, activation='relu')(dense_1)
dense_3 = Dense(128)(dense_2)
dense_4 = Dense(32)(dense_3)
output_1 = Dense(3, activation='softmax')(dense_4)
model = Model(inputs = input_1, outputs = output_1)


scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


es = EarlyStopping(monitor='loss', patience=25, mode='min', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=500, verbose=1, validation_split=1/3, shuffle=True, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# y_pred = model.predict(x_test[-5:])
# print(y_pred, '\n', y_test[-5:])

'''
[Best Fit]
with MaxAbsScaler, Tree 13 > 64 > 256(relu) > 32 > 3
batch_size=1, epochs=200, patience=25

loss =  0.19691893458366394 , accuracy =  0.9555555582046509

with No Scaler, Tree 13 > 64 > 256 > 32 > 3
batch_size=1, epochs=200, patience=25

loss =  0.5023733377456665 , accuracy =  0.9111111164093018
'''