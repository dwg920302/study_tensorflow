from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# Data
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=7/8, shuffle=True, random_state=86)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
# model = load_model('../_save/keras47_mcp.h5')    # model
model = load_model('../_save/model_checkpoint/keras47_mcp.hdf5') #   Checkpoint

start_time = time.time()
'''
model.compile(loss='categorical_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='../_save/model_checkpoint/keras47_mcp.hdf5')

hist = model.fit(x_train, y_train, epochs=25, batch_size=64, verbose=2,
    validation_split=0.05, callbacks=[es, cp])

model.save('../_save/keras47_mcp.h5')
'''

elapsed_time = time.time() - start_time

loss = model.evaluate(x_test, y_test, batch_size=64)
print('time : ', elapsed_time)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R^2 Score = ', r2)

'''
[Best Fit]
'''
