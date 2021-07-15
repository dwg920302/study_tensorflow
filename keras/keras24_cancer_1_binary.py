import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from matplotlib import pyplot as plt

dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape, y.shape)

input_1 = Input(shape=(30, ))
dense_1 = Dense(128, activation='relu')(input_1)
dense_2 = Dense(256)(dense_1)
dense_3 = Dense(512)(dense_2)
dense_4 = Dense(1024, activation='selu')(dense_3)
dense_5 = Dense(256)(dense_4)
dense_6 = Dense(128)(dense_5)
dense_7 = Dense(32, activation='elu')(dense_6)
output_1 = Dense(1, activation='sigmoid')(dense_7)
model = Model(inputs = input_1, outputs = output_1)

# -----

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=56)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=1/19, shuffle=True, callbacks=[es])

loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

y_pred = model.predict(x_test[-5:])
print(y_pred, '\n', y_test[-5:])

'''
[Best Fit]
with No Scaler, Model 30 > 128(selu) > 256 > 512(relu) > 256 > 64(elu) > 1(sigmoid)
batch_size=64, epochs=100, patience=10, loss='binary_crossentropy'
loss =  0.07363291084766388 , accuracy =  0.9655172228813171

[Better Fit]

'''

'''
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
 '''


