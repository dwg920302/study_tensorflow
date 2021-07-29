from numpy.lib.arraypad import pad
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.layers import LSTM, Bidirectional


docs = ['너무 재밋어요', '참 최고예요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고 싶네요', '글세요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요']


# 긍정 1, 부정 2
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5)
# padding = pre일 경우 앞에 0을 채우고, post일 경우 뒤에 0을 채움

print(pad_x)
print(pad_x.shape)      # (13, 5)

word_size = len(token.word_index)
print(word_size)        # 27

print(np.unique(pad_x))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# 1. 모델 구성
model = Sequential()

model.add(Embedding(input_dim = 28, output_dim=11, input_length=5))
# model.add(Embedding(28, 11))
# model.add(Embedding(28, 11, input_length=5))

model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
bidirectional (Bidirectional (None, 64)                11264
_________________________________________________________________
dense (Dense)                (None, 1)                 65
=================================================================
Total params: 11,637
Trainable params: 11,637
Non-trainable params: 0
_________________________________________________________________
PS D:\study> 
'''

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=100, batch_size=10)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ',  acc)