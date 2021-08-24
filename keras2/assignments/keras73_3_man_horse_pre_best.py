from icecream import ic

import numpy as np
import pandas as pd

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAvgPool2D, Conv2D

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=0.5,
#     shear_range=0.5,
#     fill_mode='nearest'
# )

# # train 내부 폴더 -> y의 라벨로 자동 지정됨
# # x, y 자동 생성
# data = datagen.flow_from_directory(
#     '../_data/horse_or_human',
#     target_size=(128, 128),
#     batch_size=5000,
#     class_mode='categorical',
#     shuffle=True
# )
# # Found 3309 images belonging to 2 classes.

# ic(data)

# data_x = data[0][0] # 이미지파일 같음, 데이터 할당하는데 은근 오래 걸림 (2~3분 정도)
# data_y = data[0][1] # 인덱스 같음 (남=0 여=1)

# test_datagen = ImageDataGenerator(rescale=1./255)

# np.save('../_save/_npy/k59_x_data_horsehuman_128.npy', arr=data_x)
# np.save('../_save/_npy/k59_y_data_horsehuman_128.npy', arr=data_y)

data_x = np.load('../_save/_npy/k59_x_data_horsehuman_128.npy')
data_y = np.load('../_save/_npy/k59_y_data_horsehuman_128.npy')

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=91)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def model_1(pre_model):
        model = Sequential()
        model.add(pre_model)
        model.add(Flatten())
        model.add(Dropout(1/2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(1/2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(1/2))
        model.add(Dense(2, activation='softmax'))
        return model

def model_2(pre_model):
        model = Sequential()
        model.add(pre_model)
        model.add(GlobalAvgPool2D())
        model.add(Dropout(1/2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(1/2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(1/2))
        model.add(Dense(2, activation='softmax'))
        return model

trainables = [True, False]

model_names = [[model_1, 'Flatten'], [model_2, 'GlobalAvgPool']]

es = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

for trainable in trainables:
        for loop in model_names:
                model = loop[0]
                bc = loop[1]

                pre_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
                ad = ''
                
                if trainable == True:
                        pre_model.trainable = True
                        ad = 'Trainable'
                        model = model_1(pre_model)
                else:
                        pre_model.trainable = False
                        ad = 'Non-Trainable'
                        model = model_2(pre_model)

                model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
                model.fit(x_train, y_train, batch_size=128, epochs=25, verbose=1, validation_split=1/8, shuffle=True, callbacks=es)

                loss = model.evaluate(x_test, y_test)
                print('[Condition : ', ad, ' ', bc, ']')
                print('loss = ', loss[0])
                print('accuracy = ', loss[1])

'''
[Condition :  Trainable   Flatten ]
loss =  0.6923101544380188
accuracy =  0.4805825352668762

[Condition :  Trainable   GlobalAvgPool ]
loss =  0.5069904327392578
accuracy =  0.762135922908783

[Condition :  Non-Trainable   Flatten ]
loss =  0.046642519533634186
accuracy =  0.9854369163513184

[Condition :  Non-Trainable   GlobalAvgPool ]
loss =  0.050051331520080566
accuracy =  0.9854369163513184
'''