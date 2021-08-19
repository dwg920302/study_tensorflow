from icecream import ic
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('../_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

ic(dataset.shape, dataset.describe)

dataset = dataset.values

ic(type(dataset))

x = dataset[:, :11]
y = dataset[:, 11]

for i in range(len(y)):
    if y[i] == 9.0:
        y[i] = 8.0
    if y[i] == 8.0:
        y[i] = 7.0
    if y[i] == 3.0:
        y[i] = 4.0
    if y[i] == 4.0:
        y[i] = 5.0
    # if y[i] == 5.0:
    #     y[i] = 6.0
    if y[i] == 7.0:
        y[i] = 6.0
# 또다른 해결 방법. 9를 8로 합침

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=37, stratify=y
)

model_1 = XGBClassifier(n_jobs=-1)

model_1.fit(x_train, y_train, eval_metric='mlogloss')

ic(pd.Series(y_train).value_counts())

score = model_1.score(x_test, y_test)
ic(score)   # ic| score: 0.6653061224489796


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

# 7, 8, 9 를 7로 줄였을 경우
# ic| score: 0.6605442176870748
# ic| score: 0.6476190476190476

# [7, 8, 9] 를 7, [3, 4]를 4로 줄였을 경우
# ic| score: 0.6659863945578232
# ic| score: 0.6523809523809524

# [7, 8, 9] 를 7, [3, 4, 5]를 5로 줄였을 경우
# ic| score: 0.7129251700680272
# score: 0.6965986394557823

# [7, 8, 9] 를 7, [3, 4, 5, 6]을 6으로 줄였을 경우
# ic| score: 0.8673469387755102
# ic| score: 0.8476190476190476

# [6, 7, 8, 9] 를 6, [3, 4, 5]를 5으로 줄였을 경우
# ic| score: 0.827891156462585
# ic| score: 0.8285714285714286