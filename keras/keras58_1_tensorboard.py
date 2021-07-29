import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

#* 데이터 전처리

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

'''
ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()
'''

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=40, kernel_size=(2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), activation='relu'))             
model.add(Conv2D(5, (2,2), activation='relu'))               
model.add(MaxPool2D())
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(MaxPool2D())                                                                
model.add(Flatten())                                        
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)
tb = TensorBoard(log_dir='../_save/_graph', histogram_freq=0, write_graph=True,
                write_images=True)

# 저 tb는 출력 이후 콘솔에서 tensorboard --logdir=.   커맨드로 열 수 있음

hist = model.fit(x_train, y_train, epochs=100, batch_size=1000, callbacks=[es, tb], validation_split=0.2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])