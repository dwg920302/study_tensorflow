from icecream import ic
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('../_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

ic(dataset.shape, dataset.describe)

dataset = dataset.values

ic(type(dataset))

x = dataset[:, :11].astype('float32')
y = dataset[:, 11].astype('float32')

for i in range(len(y)):
    if y[i] == 9:
        y[i] = 8
    if y[i] == 8:
        y[i] = 7
    if y[i] == 3:
        y[i] = 4
    if y[i] == 4:
        y[i] = 5

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=37, stratify=y
)

model_1 = XGBClassifier(n_jobs=-1)

model_1.fit(x_train, y_train, eval_metric='mlogloss')

ic(pd.Series(y_train).value_counts())

score = model_1.score(x_test, y_test)
ic(score)   # ic| score: 0.6653061224489796

y_pred = model_1.predict(x_test)
f1score = f1_score(y_test, y_pred, average='macro')
ic(f1score)


# Smote로 데이터 증폭 후 결과 비교 (value 별로 개수를 맞춰주기)

smote = SMOTE(random_state=37)
# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
# k_neighbors를 줄이면 해결됨
# k_neighbors = default 5. n_neighbors = k_n + 1

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

ic(pd.Series(y_smote_train).value_counts())

ic(x_smote_train.shape, y_smote_train.shape)    # ic| x_smote_train.shape: (105, 13), y_smote_train.shape: (105,)

model_2 = XGBClassifier(n_jobs=-1)

model_2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model_2.score(x_test, y_test)
ic(score)   # ic| score: 0.6353741496598639

y_pred = model_2.predict(x_test).astype('float32')
f1score = f1_score(y_test, y_pred, average='macro')
ic(f1score)


'''
ic| pd.Series(y_train).value_counts(): 6.0    1538
                                       5.0    1148
                                       7.0     742
                                       dtype: int64
ic| score: 0.7129251700680272
ic| f1score: 0.7057516118670127
ic| pd.Series(y_smote_train).value_counts(): 6.0    1538
                                             5.0    1538
                                             7.0    1538
                                             dtype: int64
ic| x_smote_train.shape: (4614, 11), y_smote_train.shape: (4614,)
ic| score: 0.6965986394557823
ic| f1score: 0.6943910669842026
'''