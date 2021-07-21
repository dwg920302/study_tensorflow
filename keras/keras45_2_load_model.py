from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
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


model = load_model('./_save/keras45_1_save_model.h5')
model.summary()

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
Epoch 25/25
891/891 - 6s - loss: 0.0158 - accuracy: 0.9952 - val_loss: 0.0257 - val_accuracy: 0.9950
Epoch 00025: early stopping
157/157 [==============================] - 0s 3ms/step - loss: 0.0214 - accuracy: 0.9935
time :  159.4049813747406
loss :  0.021443964913487434
accuracy :  0.9934999942779541
'''
