from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, GlobalAvgPool2D, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from icecream import ic
import time

from PIL import Image
from tensorflow.python.keras.saving.save import load_model


datagen = ImageDataGenerator(
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

# train 내부 폴더 -> y의 라벨로 자동 지정됨
# x, y 자동 생성
data = datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(256, 256),
    batch_size=5000,
    class_mode='binary',
    shuffle=True
)
# Found 3309 images belonging to 2 classes.

ic(data)

start_time = time.time()

# data_x = data[0][0] # 이미지파일 같음, 데이터 할당하는데 은근 오래 걸림 (2~3분 정도)
# data_y = data[0][1] # 인덱스 같음 (남=0 여=1)

# np.save('../_save/_npy/k59_x_data_menwomen.npy', arr=data_x)
# np.save('../_save/_npy/k59_y_data_menwomen.npy', arr=data_y)

data_x = np.load('../_save/_npy/k59_x_data_menwomen.npy')
data_y = np.load('../_save/_npy/k59_y_data_menwomen.npy')

elapsed_time_get_data = time.time() - start_time

ic(elapsed_time_get_data)   
ic(data_x.shape, data_y.shape)

# ic| data_x.shape: (3309, 256, 256, 3), data_y.shape: (3309,)

# men_data = datagen.flow_from_directory(
#     '../_data/men_women/men',
#     target_size=(256, 256),
#     batch_size=5000,
#     class_mode='binary',
#     shuffle=True
# )
# Found 3309 images belonging to 2 classes.

# train/test가 나뉘어 있지 않으므로 나누어 주도록 함.

# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, shuffle=True, random_state=24)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=91)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

'''
ic| x_train.shape: (2481, 256, 256, 3)
    x_test.shape: (828, 256, 256, 3)
    y_train.shape: (2481,)
    y_test.shape: (828,)
'''

ic(x_train[:5], x_train[-5:])
ic(y_train[:5], y_train[-5:])

# model = Sequential()
# model.add(Conv2D(128, (1, 1), input_shape=(256, 256, 3), padding='same'))   # 256 하니까 계속 터짐 ㅡㅡ
# model.add(MaxPool2D())
# model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
# model.add(GlobalAvgPool2D())
# model.add(Dense(256))
# model.add(Dropout(1/4))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# # model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Dense(8))
# # model.add(Dropout(0.5))
# model.add(Dense(4))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# hist = model.fit_generator(xy_train, validation_data=xy_test, validation_steps=5, epochs=10)

start_time = time.time()

# hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), validation_steps=5, epochs=10)

# model.fit(x_train, y_train, validation_split=1/4, epochs=25, shuffle=True)

# model.save('../_save/k59_5_save_model.h5') # Fit 까지 Save. 이후에 이 모델을 적용하면 그대로 평가에 쓸 수 있음.

model = load_model('../_save/k59_5_save_model.h5')

elapsed_time = time.time() - start_time

ic(elapsed_time)

# acc = max(hist.history['accuracy'])
# val_acc = max(hist.history['val_accuracy'])
# loss = max(hist.history['loss'])
# val_loss = max(hist.history['val_loss'])
# ic(acc, loss, val_acc, val_loss)

# Evaluate and Predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# 직접 사진 넣어봐야 함, 그 사진으로 predict 만들기
img = Image.open('../_data/pred_myface.jpg')
# img = Image.open('../_data/pred_woman_iu.jpg')
# img = Image.open('../_data/pred_man_uain.jpg')

img = img.resize((256, 256), Image.NEAREST)

scaler = MinMaxScaler()
x_pred = np.asarray(img).reshape(256 * 256, 3) # 사진 배열화 및 스케일링을 위해 재배열
x_pred = scaler.fit_transform(x_pred).reshape(1, 256, 256, 3)

y_acc = model.predict(x_pred)[0][0]
y_pred = round(y_acc)
if y_pred > 0:
    y_pred = '여성'
else:
    y_pred = '남성'
print('판정 결과 -> ', y_pred, ' (', y_acc, ')')


'''
[Best Fit]
ic| elapsed_time: 386.3071291446686
21/21 [==============================] - 1s 48ms/step - loss: 0.6211 - accuracy: 0.6767
loss :  0.6211317181587219
accuracy :  0.6767371892929077

[Better Fit]
ic| elapsed_time: 120.58949303627014
21/21 [==============================] - 1s 29ms/step - loss: 0.6461 - accuracy: 0.6616
loss :  0.6461209654808044
accuracy :  0.6616314053535461
'''

'''
WARNING:tensorflow:Your input ran out of data; interrupting training.
Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 25 batches).
You may need to use the repeat() function when building your dataset.
'''

'''
[성별 판정 결과 [남: 0, 여: 1]]
21/21 [==============================] - 2s 37ms/step - loss: 0.6461 - accuracy: 0.6616
loss :  0.6461209654808044
accuracy :  0.6616314053535461
판정 결과 ->  남성  ( 0.49020836 )  
'''
