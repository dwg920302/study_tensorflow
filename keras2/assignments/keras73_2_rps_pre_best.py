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


test_datagen = ImageDataGenerator(rescale=1./255)

data_x = np.load('../_save/_npy/k59_x_data_rps.npy')
data_y = np.load('../_save/_npy/k59_y_data_rps.npy')

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
        model.add(Dense(3, activation='softmax'))
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
        model.add(Dense(3, activation='softmax'))
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
loss =  1.0988965034484863
accuracy =  0.3115079402923584

[Condition :  Trainable   GlobalAvgPool ]
loss =  1.0983933210372925
accuracy =  0.3392857015132904

[Condition :  Non-Trainable   Flatten ]
loss =  0.6981768012046814
accuracy =  0.7519841194152832

[Condition :  Non-Trainable   GlobalAvgPool ]
loss =  1.0977965593338013
accuracy =  0.3492063581943512
'''