from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

from icecream import ic


# model_ls = [VGG16(), VGG19(), Xception()]

# model_ls = [ResNet50(), ResNet50V2()]

# model_ls = [ResNet101()]

# model_ls = [ResNet101V2()]

# model_ls = [ResNet152()]

# model_ls = [ResNet152V2()]

# model_ls = [DenseNet121()]

# model_ls = [DenseNet169()]

# model_ls = [DenseNet201()]

# model_ls = [InceptionV3()]

# model_ls = [InceptionResNetV2()]

# model_ls = [MobileNet(), MobileNetV2(), MobileNetV3Small()]

# model_ls = [MobileNetV3Large()]

# model_ls = [NASNetLarge()]
 
# model_ls = [NASNetMobile(), EfficientNetB0()]

# model_ls = [EfficientNetB1()]

model_ls = [EfficientNetB7()]

for model in model_ls:
    model.summary()
    ic(len(model.weights), len(model.trainable_weights))
    print('==================================================================================================')

'''
Model: "vgg16"
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
ic| len(model.weights): 32, len(model.trainable_weights): 32
==============================================================================================
Model: "vgg19"
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
ic| len(model.weights): 38, len(model.trainable_weights): 38
==============================================================================================
Model: "xception"
Total params: 22,910,480
Trainable params: 22,855,952
Non-trainable params: 54,528
__________________________________________________________________________________________________
ic| len(model.weights): 236, len(model.trainable_weights): 156
==============================================================================================
Model: "resnet50"
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________
ic| len(model.weights): 320, len(model.trainable_weights): 214
==============================================================================================
Model: "resnet50v2"
Total params: 25,613,800
Trainable params: 25,568,360
Non-trainable params: 45,440
__________________________________________________________________________________________________
ic| len(model.weights): 272, len(model.trainable_weights): 174
==============================================================================================
Model: "resnet101"
Total params: 44,707,176
Trainable params: 44,601,832
Non-trainable params: 105,344
__________________________________________________________________________________________________
ic| len(model.weights): 626, len(model.trainable_weights): 418
==================================================================================================
Model: "resnet101v2"
Total params: 44,675,560
Trainable params: 44,577,896
Non-trainable params: 97,664
__________________________________________________________________________________________________
ic| len(model.weights): 544, len(model.trainable_weights): 344
==================================================================================================
Model: "resnet152"
Total params: 60,419,944
Trainable params: 60,268,520
Non-trainable params: 151,424
__________________________________________________________________________________________________
ic| len(model.weights): 932, len(model.trainable_weights): 622
==================================================================================================
Model: "resnet152v2"
Total params: 60,380,648
Trainable params: 60,236,904
Non-trainable params: 143,744
__________________________________________________________________________________________________
ic| len(model.weights): 816, len(model.trainable_weights): 514
==================================================================================================
Model: "densenet121"
Total params: 8,062,504
Trainable params: 7,978,856
Non-trainable params: 83,648
__________________________________________________________________________________________________
ic| len(model.weights): 606, len(model.trainable_weights): 364
==================================================================================================
Model: "densenet169"
Total params: 14,307,880
Trainable params: 14,149,480
Non-trainable params: 158,400
__________________________________________________________________________________________________
ic| len(model.weights): 846, len(model.trainable_weights): 508
==================================================================================================
Model: "densenet201"
Total params: 20,242,984
Trainable params: 20,013,928
Non-trainable params: 229,056
__________________________________________________________________________________________________
ic| len(model.weights): 1006, len(model.trainable_weights): 604
==================================================================================================
Model: "inception_v3"
Total params: 23,851,784
Trainable params: 23,817,352
Non-trainable params: 34,432
__________________________________________________________________________________________________
ic| len(model.weights): 378, len(model.trainable_weights): 190
==================================================================================================
Model : "inceptionresnetv2"
Total params: 55,873,736
Trainable params: 55,813,192
Non-trainable params: 60,544
__________________________________________________________________________________________________
ic| len(model.weights): 898, len(model.trainable_weights): 490
==================================================================================================
Model: "mobilenet_1.00_224"
Total params: 4,253,864
Trainable params: 4,231,976
Non-trainable params: 21,888
_________________________________________________________________
ic| len(model.weights): 137, len(model.trainable_weights): 83
==================================================================================================
Model: "mobilenetv2_1.00_224"
Total params: 3,538,984
Trainable params: 3,504,872
Non-trainable params: 34,112
__________________________________________________________________________________________________
ic| len(model.weights): 262, len(model.trainable_weights): 158
==================================================================================================
Model: "MobilenetV3Small"
Total params: 2,554,968
Trainable params: 2,542,856
Non-trainable params: 12,112
__________________________________________________________________________________________________
ic| len(model.weights): 210, len(model.trainable_weights): 142
==================================================================================================
Model: "MobilenetV3large"
Total params: 5,507,432
Trainable params: 5,483,032
Non-trainable params: 24,400
__________________________________________________________________________________________________
ic| len(model.weights): 266, len(model.trainable_weights): 174
==================================================================================================
Model: "NASNetLarge"
Total params: 88,949,818
Trainable params: 88,753,150
Non-trainable params: 196,668
__________________________________________________________________________________________________
ic| len(model.weights): 1546, len(model.trainable_weights): 1018
==================================================================================================
Model: "NASNetMobile"
Total params: 5,326,716
Trainable params: 5,289,978
Non-trainable params: 36,738
__________________________________________________________________________________________________
ic| len(model.weights): 1126, len(model.trainable_weights): 742
==================================================================================================
Model: "efficientnetb0"
Total params: 5,330,571
Trainable params: 5,288,548
Non-trainable params: 42,023
__________________________________________________________________________________________________
ic| len(model.weights): 314, len(model.trainable_weights): 213
==================================================================================================
Model: "efficientnetb1"
Total params: 7,856,239
Trainable params: 7,794,184
Non-trainable params: 62,055
__________________________________________________________________________________________________
ic| len(model.weights): 442, len(model.trainable_weights): 301
==================================================================================================
Model: "efficientnetb7"
Total params: 66,658,687
Trainable params: 66,347,960
Non-trainable params: 310,727
__________________________________________________________________________________________________
ic| len(model.weights): 1040, len(model.trainable_weights): 711
==================================================================================================
'''