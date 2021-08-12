from icecream import ic
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer


# (x_train, _), (x_test, _) = mnist.load_data()
# _ it's called underscore and used for ignoring the specific values and others

(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, x_test.shape)

# x = np.append(x_train, x_test, axis=0)
# y = np.append(y_train, y_test, axis=0)

# ic(x.shape)

# x_shape = x.shape

x_shape = x_train.shape

# 실습
# pca를 통해 0.95 이상인 게 몇 개?

pca = PCA()
x_train = pca.fit_transform(x_train.reshape(x_shape[0], x_shape[1]*x_shape[2]))
x_test = pca.transform(x_test.reshape(x_test.shape[0], x_shape[1]*x_shape[2]))

pca_evr = pca.explained_variance_ratio_
ic(pca_evr, sum(pca_evr))

cumsum = np.cumsum(pca_evr)
ic(cumsum) 
# ic| pca_evr: array([0.40242142, 0.14923182, 0.12059623, 0.09554764, 0.06621856,
#                     0.06027192, 0.05365605, 0.04336832, 0.00783199, 0.00085605])
#     sum(pca_evr): 1.0
# ic| cumsum: array([0.40242142, 0.55165324, 0.67224947, 0.76779711, 0.83401567,
#                    0.89428759, 0.94794364, 0.99131196, 0.99914395, 1.        ])

ic(np.argmax(cumsum >= 0.95)+1)

# ic| np.argmax(cumsum >= 0.95)+1: 154

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

# ic(np.argmax(cumsum >= 0.99)+1)

# ic| np.argmax(cumsum >= 0.99)+1: 331

# Tensorflow DNN으로 구성하고, 기존 Tensorflow DNN과 비교

n_components = np.argmax(cumsum >= 0.95)+1

pca = PCA(n_components=n_components)  # 154

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

x_train_2 = pca.fit_transform(x_train)
x_test_2 = pca.transform(x_test)

scaler = MinMaxScaler()
x_train_2 = scaler.fit_transform(x_train_2)
x_test_2 = scaler.transform(x_test_2)

# Model
model = Sequential()
model.add(Dense(128, input_shape=(n_components, )))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile and fit

es = EarlyStopping(monitor='val_accuracy', patience=25, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train_2, y_train, batch_size=64, epochs=100, verbose=2, validation_split=1/12, callbacks=[es])

# evaluate

loss = model.evaluate(x_test_2, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

'''
그냥 CNN
loss =  0.10423552244901657 , accuracy =  0.9807000160217285

PCA 0.95의 CNN
loss =  0.08252689987421036 , accuracy =  0.9797000288963318
'''