from tensorflow.python.keras.saving.save import load_model
from modellaboratory import use_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from icecream import ic
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt, font_manager, rc
from sklearn.metrics import r2_score

x1 = pd.DataFrame(np.load('../_save/X1_COVID_KOR_pp.npy', allow_pickle=True))
x2 = pd.DataFrame(np.load('../_save/X2_COVID_KOR_pp.npy', allow_pickle=True))
y = pd.DataFrame(np.load('../_save/Y_COVID_KOR_pp.npy', allow_pickle=True))

ic(x1.shape, x2.shape, y.shape)  # ic| x1.shape: (367, 5), x2.shape: (367, 13), y.shape: (366, 1)

# pred 골라내기 (가장 최신 데이터 -> 그 다음 날 확진자)

x1_pred = x1[-1:]
x2_pred = x2[-1:]
x1 = x1[:-1]
x2 = x2[:-1]

ic(x1.head(), x2.head())

ic(x1.shape, x1.head())

# 데이터 증폭

x1 = x1.to_numpy()
x2 = x2.to_numpy()
y = y.to_numpy()

x1_augment = x1.copy()
x2_augment = x2.copy()
y_augment = y.copy()

amplify_size = 3    # 이 수치를 수정하여 데이터를 얼마나 반복시킬 지 정함

for i in range(amplify_size):
    x1 = np.concatenate((x1, x1_augment))
    x2 = np.concatenate((x2, x2_augment))
    y = np.concatenate((y, y_augment))

model = use_model(x1.shape[1], x2.shape[1])

# model = load_model('../_save/covid_model.h5')

model.summary()

# for LSTM, GRU
# x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
# x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# model.compile(optimizer='adam', loss='mse', run_eagerly=True)

model.compile(optimizer='adam', loss='mse')

start_time = time()

es = EarlyStopping(patience=50, verbose=1, restore_best_weights=True)

chkpt = ModelCheckpoint('../_save/model_checkpoint/covid_checkpoint.hdf5')

history = model.fit([x1, x2], y, verbose=2, epochs=500, batch_size=16, validation_split=1/6, validation_batch_size=16, 
            callbacks=[es, chkpt])

history = history.history

# history = model.fit([x1, x2], y, verbose=2, epochs=1000, batch_size=1, validation_split=0.05, validation_batch_size=1, 
#               callbacks=[EarlyStopping(patience=100, verbose=1, restore_best_weights=True)],
#               shuffle=True)

elapsed_time = time() - start_time
print('소요시간 = ', elapsed_time)

y_pred = model.predict([x1_pred, x2_pred])
ic(y_pred)

# ic(history)

# r2_score


# 길이 조정
size = 2
x1_test = x1[-(size):]
x1_test = np.concatenate((x1_test, x1_pred))
x2_test = x2[-(size):]
x2_test = np.concatenate((x2_test, x2_pred))
y_test = y[-(size):]
y_test = np.concatenate((y_test, y_pred))

# 길이 그대로
# x1_test = x1
# x2_test = x2
# y_test = y

y_r2 = model.predict([x1_test, x2_test])

r2_score = r2_score(y_test, y_r2)
ic(r2_score)

# 시각화

# 한글깨짐 해결
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.plot(history['loss'])
plt.plot(history['val_loss'])

plt.title('오차 [Train, Validation]')
plt.xlabel('시행회수')
plt.ylabel('오차(Train), 오차(Validation)')
plt.legend(['train_loss', 'val_loss'])
plt.show()

model.save('../_save/covid_model.h5')


