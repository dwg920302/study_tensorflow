from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding
from tensorflow.python.keras.layers.recurrent import LSTM

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화네요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고 싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

input_1 = Input(shape=(5, ))
hid_1 = Embedding(input_dim=28, output_dim=11, input_length=5)(input_1)
hid_2 = LSTM(32)(hid_1)
output_1 = Dense(1, activation='sigmoid')(hid_2)

model = Model(inputs=input_1, outputs=output_1)

model.summary()

'''
Epoch 100/100
13/13 [==============================] - 0s 19ms/step - loss: 1.9266e-04 - accuracy: 1.0000
1/1 [==============================] - 0s 207ms/step - loss: 2.7513e-04 - accuracy: 1.0000
1.0
'''

x = token.texts_to_sequences(docs)
print(x)

pad_x = pad_sequences(x, padding='pre', maxlen=5)
# padding = pre일 경우 앞에 0을 채우고, post일 경우 뒤에 0을 채움
print(pad_x, pad_x.shape)   # (13, 5)
print(len(np.unique(pad_x))) # (28, )

# Compile and Fit

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

# Evaulate

acc = model.evaluate(pad_x, labels)[1]
print(acc)

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,973
Trainable params: 5,973
Non-trainable params: 0
_________________________________________________________________
'''