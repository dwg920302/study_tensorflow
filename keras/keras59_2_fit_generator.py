from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from icecream import ic


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

hist = model.fit_generator(xy_train, validation_data=xy_test, validation_steps=5, epochs=10)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위를 시각화 할것

ic('acc : ', acc[-1])
ic('val_acc : ', acc[-1])

'''
ic| 'acc : ', acc[-1]: 0.6937500238418579
ic| 'val_acc : ', acc[-1]: 0.6937500238418579
'''