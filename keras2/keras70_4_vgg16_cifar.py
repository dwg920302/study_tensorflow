from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping


# Trainable = True or False 비교
# FC vs GlobalAvgPool 비교

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])).reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
x_test = scaler.transform(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])).reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3), classifier_activation='softmax')

# vgg16.trainable = False # 훈련 동결

model = Sequential()
model.add(vgg16)
model.add(GlobalAvgPool2D())
model.add(Dropout(3/8))
model.add(Dense(64, activation='relu'))
model.add(Dropout(3/8))
model.add(Dense(10, activation='softmax'))

# model = Sequential()
# model.add(vgg16)
# model.add(Flatten())
# model.add(Dropout(3/8))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(3/8))
# model.add(Dense(10, activation='softmax'))

model.summary()

es = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

# model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=25, verbose=1, validation_split=1/8, shuffle=True, callbacks=es)


loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
(trainable, fc)
Restoring model weights from the end of the best epoch.
Epoch 00017: early stopping
313/313 [==============================] - 2s 7ms/step - loss: 0.7296 - accuracy: 0.7819
loss =  0.7295808792114258
accuracy =  0.7818999886512756

(non_trainable, fc)
Epoch 25/25
342/342 [==============================] - 5s 14ms/step - loss: 1.4407 - accuracy: 0.4905 - val_loss: 1.2422 - val_accuracy: 0.5811
313/313 [==============================] - 3s 7ms/step - loss: 1.2709 - accuracy: 0.5635
loss =  1.2709094285964966
accuracy =  0.5634999871253967

(trainable, gap)
Epoch 6/25
342/342 [==============================] - 13s 38ms/step - loss: 2.3027 - accuracy: 0.0981 - val_loss: 2.3028 - val_accuracy: 0.0946
Restoring model weights from the end of the best epoch.
Epoch 00006: early stopping
313/313 [==============================] - 3s 7ms/step - loss: 2.3026 - accuracy: 0.1000
loss =  2.3026041984558105
accuracy =  0.10000000149011612

(non_trainable, gap)
Restoring model weights from the end of the best epoch.
Epoch 00023: early stopping
313/313 [==============================] - 3s 8ms/step - loss: 1.2696 - accuracy: 0.5598
loss =  1.2695564031600952
accuracy =  0.5598000288009644
'''