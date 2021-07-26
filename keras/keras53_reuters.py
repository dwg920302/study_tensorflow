from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

'''
이 데이터는 단어들이 몇 번 등장하는 지의 빈도에 따라서 인덱스를 부여했습니다.
1이라는 숫자는 이 단어가 이 데이터에서 등장 빈도가 1등이라는 뜻입니다.
27,595라는 숫자는 이 단어가 데이터에서 27,595번째로 빈도수가 높은 단어라는 뜻입니다.
즉, 실제로는 빈도가 굉장히 낮은 단어라는 뜻입니다.

앞서 num_words에다가 None을 부여했는데, 만약 num_words에 1,000을 넣었다면
빈도수 순위가 1,000 이하의 단어만 가져온다는 의미이므로 데이터에서 1,000을 넘는 정수는 나오지 않습니다.
'''

ic(x_train[0], type(x_train[0]))
ic(x_train[1], type(x_train[1]))
ic(len(x_train[0]), len(x_train[1]))
ic(y_train[0], type(y_train[0]))
ic(y_train[1], type(y_train[1]))

ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

'''
ic| x_train.shape: (8982,)
    y_train.shape: (8982,)
    x_test.shape: (2246,)
    y_test.shape: (2246,)
'''


print(type(x_train))    # class 'numpy.ndarray

print("뉴스 기사의 최대 길이 : ", max(len(i) for i in x_train))  # 뉴스 기사의 최대 길이 :  2376
print("뉴스 기사의 평균 길이 : ", sum(map(len, x_train)) / len(x_train))  # 뉴스 기사의 평균 길이 : 145.5

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

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# Model

ic(x_train.shape, x_test.shape)

model = Sequential()
model.add(Embedding(input_dim=100, output_dim=32))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

model.summary()

# Compile and Fit

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# Evaulate

acc = model.evaluate(x_test, y_test)[1]
print(acc)