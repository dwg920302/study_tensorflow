from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

from tensorflow.keras.datasets import cifar10

import pandas as pd
from icecream import ic

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# vgg16.trainable = False # 훈련 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(1))

model.summary()

ic(len(model.weights))
ic(len(model.trainable_weights))

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
res = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Trainable Layer'])
ic(res)

'''
ic| res:                                                                             Layer Type Layer Name  Trainable Layer
         0  <tensorflow.python.keras.engine.functional.Functional object at 0x0000025FC3148AC0>  vgg16      True
         1  <tensorflow.python.keras.layers.core.Flatten object at 0x0000025FC3143DF0>           flatten    True
         2  <tensorflow.python.keras.layers.core.Dense object at 0x0000025FC313CB20>             dense      True
         3  <tensorflow.python.keras.layers.core.Dense object at 0x0000025FC31782B0>             dense_1    True
'''
'''
ic| res:                                                                             Layer Type Layer Name  Trainable Layer
         0  <tensorflow.python.keras.engine.functional.Functional object at 0x000002212A1BAAC0>  vgg16      False
         1  <tensorflow.python.keras.layers.core.Flatten object at 0x000002212A1B5970>           flatten    True
         2  <tensorflow.python.keras.layers.core.Dense object at 0x000002212A215730>             dense      True
'''