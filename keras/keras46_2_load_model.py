from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from icecream import ic
import time

# 1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model

# model = load_model('./_save/keras46_1_save_model_1.h5')
model = load_model('./_save/keras46_1_save_model_2.h5')
model.summary()
# 3. compile fit
start_time = time.time()
'''
model.compile(loss='categorical_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2,
    validation_split=0.05, callbacks=[es])
'''

elapsed_time = time.time() - start_time

# 4. predict eval -> no need to

y_pred = model.predict([x_test])

loss = model.evaluate(x_test, y_test)
print("elapsed time = ", elapsed_time)

print('loss : ', loss)

r2 = r2_score(y_test, y_pred)
print('R^2 score : ', r2)


'''
[Best Fit]
'''
