# test 데이터를 기존 대비 +20%
# 성과 비교
# 증폭 데이터는 temp에 저장 후 결과 본 뒤 삭제

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from icecream import ic


test_datagen = ImageDataGenerator(rescale=1./255)

data_x = np.load('../_save/_npy/k59_x_data_menwomen.npy')
data_y = np.load('../_save/_npy/k59_y_data_menwomen.npy')

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=91)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

augment_size=int(x_test.shape[0]/5)

randidx=np.random.randint(augment_size, size=augment_size)
ic(randidx, randidx.shape)

x_augmented = x_test[randidx].copy()
y_augmented = y_test[randidx].copy()

x_augmented = test_datagen.flow(x_augmented,
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False,
                                save_to_dir='../temp/').next()[0]

x_test = np.concatenate((x_test, x_augmented))
y_test = np.concatenate((y_test, y_augmented))

ic(x_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(256, 256, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

acc = max(hist.history['accuracy'])
val_acc = max(hist.history['val_accuracy'])
loss = min(hist.history['loss'])
val_loss = min(hist.history['val_loss'])

ic(loss, acc, val_loss, val_acc)

'''
ic| loss: 0.15915653109550476
    acc: 0.9478654861450195
    val_loss: 1.0417815446853638
    val_acc: 0.5869017839431763
    '''