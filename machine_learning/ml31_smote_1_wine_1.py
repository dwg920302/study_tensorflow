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

dataset = load_wine()
x = dataset.data
y = dataset.target

ic(x.shape, y.shape)    # ic| x.shape: (178, 13), y.shape: (178,) 

ic(pd.Series(y).value_counts())

# ic| pd.Series(y).value_counts(): 1    71
#                                  0    59
#                                  2    48
#                                  dtype: int64

x_new = x
y_new = y
ic(x_new.shape, y_new.shape)
ic(pd.Series(y_new).value_counts())

# ic| pd.Series(y_new).value_counts(): 1    71
#                                      0    59
#                                      2    18
#                                      dtype: int64

x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, train_size=0.7, shuffle=True, random_state=37, stratify=y_new
)

ic(pd.Series(y_train).value_counts())

# ic| pd.Series(y_train).value_counts(): 1    35
#                                        0    30
#                                        2     9
#                                        dtype: int64

# stratify를 설정할 경우 랜덤 스테이트에 관계없이 같은 비율로 데이터를 깎음.


model_1 = XGBClassifier(n_jobs=-1)

model_1.fit(x_train, y_train, eval_metric='mlogloss')

score = model_1.score(x_test, y_test)
ic(score)   # ic| score: 0.9629629629629629


# Smote로 데이터 증폭 후 결과 비교 (value 별로 개수를 맞춰주기)

smote = SMOTE(random_state=37)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

ic(pd.Series(y_smote_train).value_counts())

# ic| pd.Series(y_smote_train).value_counts(): 1    35
#                                              0    35
#                                              2    35
#                                              dtype: int64

# value가 같은 수로 맞춰졌음

ic(x_smote_train.shape, y_smote_train.shape)    # ic| x_smote_train.shape: (105, 13), y_smote_train.shape: (105,)

model_2 = XGBClassifier(n_jobs=-1)

model_2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model_2.score(x_test, y_test)
ic(score)   # ic| score: 0.9629629629629629

