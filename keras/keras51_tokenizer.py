from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

# 3 1 4 5 6 1 2 2 7

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

'''
{'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}

정렬 기준 : 나온 숫자 (진짜, 마구는 2번 나왔으므로 앞 순서), 그 이후에는 해당 문자가 처음으로 등장한 순서
'''

# 저렇게 정렬된 값이 진짜 저렇게 가치의 등급이 나뉘어지는 게 아니므로, One Hot Vector (Encode), categorical이 필요함.

x = token.texts_to_sequences([text])
print(x)

# [[3, 1, 4, 5, 6, 1, 2, 2, 7]]

word_size = len(token.word_index)

x1 = to_categorical(x)

print(x1, x1.shape)

'''
[[3, 1, 4, 5, 6, 1, 2, 2, 7]]
[[[0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1.]]]
  
  (1, 9, 8)
'''