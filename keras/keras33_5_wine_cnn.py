import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalAvgPool1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer
from sklearn.preprocessing import OneHotEncoder


dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)
print(dataset)

print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# wine의 quailty를 y로 잡음

y = dataset['quality']
x = dataset.drop(columns='quality')

print(x, y)
print(x.shape, y.shape)

print(y.unique()) # 7종류 [3,4,5,6,7,8,9]

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.755, shuffle=True, random_state=23)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(x_train.shape[0], 11, 1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0], 11, 1)

# Model

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=(1, ),                   
                        padding='same', 
                        input_shape=(11, 1)))
model.add(Conv1D(32, (1, ), padding='same'))
model.add(Conv1D(128, (1, ), padding='same'))
model.add(Dropout(0.1))
model.add(Conv1D(256, (1, ), padding='same', activation='relu'))               
model.add(Conv1D(32, (1, ), padding='same'))
# model.add(MaxPool1D())
model.add(GlobalAvgPool1D())
model.add(Dense(7, activation='linear'))
    
# es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

es = EarlyStopping(monitor='accuracy', patience=30, mode='max', verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=2, shuffle=True, validation_split=1/151, callbacks=[es])
# ValueError: Shapes (None, 7) and (None, 11, 7) are incompatible

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

# 6.025832176208496 , accuracy =  0.028309741988778114