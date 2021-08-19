# Smote (Synthetic Minority Oversampling Technique, 합성 소수 오버샘플링 기법)

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

ic(x.shape, y.shape)    # ic| x.shape: (4898, 11), y.shape: (4898,) 

ic(pd.Series(y).value_counts())

# ic| pd.Series(y).value_counts(): 6.0    2198
#                                  5.0    1457
#                                  7.0     880
#                                  8.0     175
#                                  4.0     163
#                                  3.0      20
#                                  9.0       5

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=37, stratify=y
)

ic(pd.Series(y_train).value_counts())

# ic| pd.Series(y_train).value_counts(): 6.0    1538
#                                        5.0    1020
#                                        7.0     616
#                                        8.0     122
#                                        4.0     114
#                                        3.0      14
#                                        9.0       4
#                                        dtype: int64

# stratify를 설정할 경우 랜덤 스테이트에 관계없이 같은 비율로 데이터를 깎음.


model_1 = XGBClassifier(n_jobs=-1)

model_1.fit(x_train, y_train, eval_metric='mlogloss')

score = model_1.score(x_test, y_test)
ic(score)   # ic| score: 0.6653061224489796


# Smote로 데이터 증폭 후 결과 비교 (value 별로 개수를 맞춰주기)

smote = SMOTE(random_state=3, k_neighbors=3)

# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
# k_neighbors를 줄이면 해결됨
# k_neighbors = default 5. n_neighbors = k_n + 1



x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

ic(pd.Series(y_smote_train).value_counts())

ic(x_smote_train.shape, y_smote_train.shape)    # ic| x_smote_train.shape: (105, 13), y_smote_train.shape: (105,)

model_2 = XGBClassifier(n_jobs=-1)

model_2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model_2.score(x_test, y_test)
ic(score)   # ic| score: 0.6414965986394557

