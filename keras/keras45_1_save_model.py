from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from icecream import ic
import time

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 2. model


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2),                          
                        padding='valid', activation='relu', 
                        input_shape=(28, 28, 1)))
model.add(Dropout(0.15))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
model.add(Dropout(0.15))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Dropout(0.15))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))

model.add(GlobalAvgPool2D())
# model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
# model.save('../_save/keras45_1_save_model.h5')
model.save('./_save/keras45_1_save_model.h5') # 현재까지의 진행상황이 저장된 모델을 Save

# 3. compile fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2,
    validation_split=0.05, callbacks=[es])
elapsed_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=64)
print('time : ', elapsed_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
[Best Fit]
Epoch 00044: early stopping
157/157 [==============================] - 1s 3ms/step - loss: 0.0258 - accuracy: 0.9942
time :  280.96207904815674
loss[category] :  0.025827858597040176
loss[accuracy] :  0.9941999912261963
'''
