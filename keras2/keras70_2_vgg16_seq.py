from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

from tensorflow.keras.datasets import cifar10

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# vgg16.trainable = False # 훈련 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 7, 7, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 32)                802848
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33
=================================================================
Total params: 15,517,569
Trainable params: 15,517,569
Non-trainable params: 0
_________________________________________________________________
'''
'''
Model: "sequential" with non_trainable
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 7, 7, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 32)                802848
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33
=================================================================
Total params: 15,517,569
Trainable params: 802,881
Non-trainable params: 14,714,688
_________________________________________________________________
'''