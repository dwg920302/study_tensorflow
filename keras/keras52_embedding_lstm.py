from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers.recurrent import LSTM

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화네요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고 싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

'''
{'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화네요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 
21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}
'''

x = token.texts_to_sequences(docs)
print(x)

'''
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
'''

# 0을 채워야 함

pad_x = pad_sequences(x, padding='pre', maxlen=5)
# padding = pre일 경우 앞에 0을 채우고, post일 경우 뒤에 0을 채움
print(pad_x, pad_x.shape)   # (13, 5)
print(len(np.unique(pad_x))) # (28, )

'''
[[ 0  0  0  2  4]
 [ 0  0  0  1  5]
 [ 0  1  3  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]
 [ 0 25  3 26 27]] (13, 5)
'''

pad_x1 = pad_sequences(x, padding='pre', maxlen=4)
print(pad_x1, pad_x1.shape)

'''
[[ 0  0  2  4]
 [ 0  0  1  5]
 [ 1  3  6  7]
 [ 0  8  9 10]
 [12 13 14 15]
 [ 0  0  0 16]
 [ 0  0  0 17]
 [ 0  0 18 19]
 [ 0  0 20 21]
 [ 0  0  0 22]
 [ 0  0  2 23]
 [ 0  0  1 24]
 [25  3 26 27]] (13, 4)
'''

pad_x2 = pad_sequences(x, padding='post', maxlen=4)
print(pad_x2, pad_x2.shape)

'''
[[ 2  4  0  0]
 [ 1  5  0  0]
 [ 1  3  6  7]
 [ 8  9 10  0]
 [12 13 14 15]
 [16  0  0  0]
 [17  0  0  0]
 [18 19  0  0]
 [20 21  0  0]
 [22  0  0  0]
 [ 2 23  0  0]
 [ 1 24  0  0]
 [25  3 26 27]] (13, 4)
'''


# Label의 개수만큼!

# 카테고리컬화 (OneHotEncoding) 시킬 경우 (13, 5, 27)이 됨.

# 카테고리컬화 시킬 경우, 그 종류가 많아지면 많아질수록 사이즈가 기하급수적으로 커지는 문제가 생김.

# 그래서, 카테고리컬한 값들을 벡터화 (2차원 혹은 n차원) 시킬 필요가 있음


# model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# # (input_dim = 단어사전의 갯수, output_dim = output 노드의 갯수, input_length = 한 문장의 길이)
# # 이 Embedding만 unit 이 중앙에 들어감.
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308 
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,962
Trainable params: 5,962
Non-trainable params: 0
_________________________________________________________________
297 = 11 * 27 -> input_dim * output_dim

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 77)             2156
_________________________________________________________________
lstm (LSTM)                  (None, 32)                14080
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 16,192
Trainable params: 16,192
Non-trainable params: 0
_________________________________________________________________
2079 = 27 * 77 -> input_dim * output_dim

'''

# model = Sequential()
# model.add(Embedding(28, 77))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 77)          2156
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                14080
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33
=================================================================
Total params: 16,192
Trainable params: 16,192
Non-trainable params: 0
_________________________________________________________________
-> Input dim -> Embedding Output_shape의 2번째 칸
'''

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile and Fit

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

# Evaulate

acc = model.evaluate(pad_x, labels)[1]
print(acc)

'''
Epoch 100/100
13/13 [==============================] - 0s 19ms/step - loss: 1.8083e-04 - accuracy: 1.0000
1/1 [==============================] - 0s 208ms/step - loss: 3.2328e-04 - accuracy: 1.0000
1.0
'''