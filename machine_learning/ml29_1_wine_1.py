from icecream import ic
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

dataset = pd.read_csv('../_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

ic(dataset.shape, dataset.describe)

dataset = dataset.values

ic(type(dataset))

'''
ic| dataset.shape: (4898, 12)
    dataset.describe: <bound method NDFrame.describe of       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides 
 ...  density    pH  sulphates  alcohol  quality
                      0               7.0              0.27         0.36            20.7      0.045  ...  1.00100  3.00       0.45    
  8.8        6
                      1               6.3              0.30         0.34             1.6      0.049  ...  0.99400  3.30       0.49    
  9.5        6
                      2               8.1              0.28         0.40             6.9      0.050  ...  0.99510  3.26       0.44    
 10.1        6
                      3               7.2              0.23         0.32             8.5      0.058  ...  0.99560  3.19       0.40    
  9.9        6
                      4               7.2              0.23         0.32             8.5      0.058  ...  0.99560  3.19       0.40    
  9.9        6
                      ...             ...               ...          ...             ...        ...  ...      ...   ...        ...    
  ...      ...
                      4893            6.2              0.21         0.29             1.6      0.039  ...  0.99114  3.27       0.50    
 11.2        6
                      4894            6.6              0.32         0.36             8.0      0.047  ...  0.99490  3.15       0.46    
  9.6        5
                      4895            6.5              0.24         0.19             1.2      0.041  ...  0.99254  2.99       0.46    
  9.4        6
                      4896            5.5              0.29         0.30             1.1      0.022  ...  0.98869  3.34       0.38    
 12.8        7
                      4897            6.0              0.21         0.38             0.8      0.020  ...  0.98941  3.26       0.32    
 11.8        6

                      [4898 rows x 12 columns]>
ic| type(dataset): <class 'numpy.ndarray'>
'''

x = dataset[:, :11]
y = dataset[:, 11]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11, shuffle=True, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

ic(score)       # ic| score: 0.6908163265306122