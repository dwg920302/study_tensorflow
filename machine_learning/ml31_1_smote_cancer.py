# f1 score

from icecream import ic
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

x, y = load_breast_cancer(return_X_y=True)

ic(x.shape, y.shape)
ic(x[:5], y[:5])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=48, stratify=y
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

smote = SMOTE(random_state=48)
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

y_pred = model_2.predict(x_test)
ic(type(y_pred), y_pred)
f1score = f1_score(y_test, y_pred, average='macro')
ic(f1score)