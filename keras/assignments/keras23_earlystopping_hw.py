# 어떤 조건을 만족했을 때, epoch를 중지시키는 것

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt, font_manager, rc

# 한글깨짐 해결
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x))
print(np.min(y), np.max(y))

print(x.shape)
print(y.shape)

# 데이터 전처리(preprocess)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=38)

scaler = StandardScaler()
scaler.fit(x_train)   # train만! test는 포함시키지 않고, 이 train을 fit시킨 걸로만 test함.
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train), np.min(x_test), np.max(x_test))

input_1 = Input(shape=(13, ))
dense_1 = Dense(128, activation='relu')(input_1)
dense_2 = Dense(64)(dense_1)
dense_3 = Dense(64)(dense_2)
dense_4 = Dense(32, activation='relu')(dense_3)
output_1 = Dense(1)(dense_4)
model = Model(inputs = input_1, outputs = output_1)

# model.summary()


# 컴파일 및 훈련

model.compile(loss="mse", optimizer='adam')

# es_1 = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
# es_2 = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# model.fit(x_train, y_train, batch_size=16, epochs=250, verbose=1, validation_split=1/18, shuffle=True, callbacks=[es_1, es_2])

# 위의 조건 둘 중 하나의 조건이 충족되면 스톱함. (or)

es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

hist = model.fit(x_train, y_train, batch_size=16, epochs=250, verbose=1, validation_split=1/18, shuffle=True, callbacks=[es])

# 평가(evaluate) 및 예측

loss = model.evaluate(x_test, y_test)

print('loss = ', loss)

y_pred = model.predict(x_test)

# print('예측값 = ', y_pred)

# R2 구하기

r2 = r2_score(y_test, y_pred)
print('R2 = ', r2)

# Pyplot

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('오차 [Train, Validation]')
plt.xlabel('시행회수')
plt.ylabel('오차(Train), 오차(Validation)')
plt.legend(['train_loss', 'val_loss'])
plt.show()

'''
[Best Fit]
batch_size=32, epochs=250
loss =  6.699991226196289
R2 =  0.8918041789532436

[Better Fit]

'''
