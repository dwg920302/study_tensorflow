from tensorflow.keras.datasets import imdb
from icecream import ic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 10000
)

'''
영화 사이트 IMDB의 리뷰 데이터입니다.
이 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 긍정인 경우 1을 부정인 경우 0으로 표시한 레이블로 구성된 데이터입니다.

케라스에서 제공하는 IMDB 리뷰 데이터는 앞서 배운 로이터 뉴스 데이터에서 훈련 데이터와 테스트 데이터를
우리가 직접 비율을 조절했던 것과는 달리 이미 훈련 데이터와 테스트 데이터를 50:50 비율로 구분해서 제공합니다.
로이터 뉴스 데이터에서 사용했던 test_split과 같은 데이터의 비율을 조절하는 파라미터는 imdb.load_data에서는 지원하지 않습니다.
'''

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (25000,), (25000,), (25000,), (25000,)
ic(x_train[0], x_test[0])
ic(y_train[0], y_test[0])

print(type(x_train))    # class 'numpy.ndarray

print("최대 길이 : ", max(len(i) for i in x_train))  # 최대 길이 :  2494
print("평균 길이 : ", sum(map(len, x_train)) / len(x_train))  # 평균 길이 : 238.71364

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape)  # (8982, 100) (2246, 100)
print(type(x_train), type(x_train[0]))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
ic(x_train[0])

ic(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

# Model

ic(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

# 이 부분 미완

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=16, input_length=100))
# 10000은 되는데 10000 미만의 값들은 안 됨. 왜 10000이 되는지 이유는 아직 모르겠음
model.add(LSTM(32))
model.add(Dense(2, activation='sigmoid'))

model.summary()

# Compile and Fit

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split = 0.002)

# Evaulate

res = model.evaluate(x_test, y_test)
print('loss = ', res[0])
print('accuracy = ', res[1])

'''
Epoch 10/10
780/780 [==============================] - 7s 9ms/step - loss: 0.0580 - accuracy: 0.9834 - val_loss: 0.3337 - val_accuracy: 0.8600
782/782 [==============================] - 3s 4ms/step - loss: 0.6523 - accuracy: 0.8242
loss =  0.6522886157035828
accuracy =  0.8241999745368958
'''