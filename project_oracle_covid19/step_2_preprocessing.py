from icecream import ic
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import numpy as np
import pandas as pd

x1 = pd.DataFrame(np.load('../_save/X1_COVID_KOR.npy', allow_pickle=True))
x2 = pd.DataFrame(np.load('../_save/X2_COVID_KOR.npy', allow_pickle=True))

ic(x1.shape, x2.shape)  # x1.shape: (368, 4), x2.shape: (367, 12)
# x1은 x2와 다르게 전 날의 데이터 한 줄이 추가되어 있어서 1줄 많음.

ic(x1.head(), x2.head())

columns_x1 = ['Confirmed_Total', 'Deaths_Total', 'Recovered_Total', 'Confirmed_Today']
columns_x2 = ['avg_temp_avg', 'avg_temp_min', 'avg_temp_max', 
            'tmp_diff_avg', 'tmp_diff_min', 'tmp_diff_max',
            'rainfall_avg', 'rainfall_min', 'rainfall_max',
            'humidity_avg', 'humidity_min', 'humidity_max']

# Confirmed_Total의 데이터 -> Confirmed_Today

# 시계열 데이터를 활용하여

# 오늘 발생한 확진자 수를 구함. 오늘까지 확진자 총합 - 어제까지 확진자 총합 = 오늘 확진자 수

# 오늘 확진자 수를 predict의 결과인 y값으로 함

# dataset = x1[1]
dataset = x1[0] 

def split_x(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        arr.append(subset)
    return np.array(arr)

dataset = split_x(dataset, 2)

ic(dataset.shape)

y_d1 = np.array([data[0] for data in dataset])    # 어제까지의 확진자수 총합
y_dday = np.array([data[1] for data in dataset])    # 오늘까지의 확진자수 총합

y_tmp = np.array([[]])

ic(y_d1[:5], y_dday[:5])

for i in range(y_dday.shape[0]):
    ic(y_d1[i], y_dday[i])
    y_tmp = np.append(y_tmp, str(float(y_dday[i]) - float(y_d1[i])))

ic(y_tmp, y_tmp.shape)

y = y_tmp[1:]

x1 = x1[1:] # 7월 31일 데이터 삭제 (크기를 맞추어 주어야 하므로)
x1[3] = y_tmp

ic(y.shape)

ic(x1.head(), x1.shape, x2.shape)

np.save('../_save/X1_COVID_KOR_p.npy', arr=x1)

# It's Scaler Time!

# date_scaler = MinMaxScaler()
# scalerset_x1 = [date_scaler]

# for i in range(x1.shape[1] - 1):
#     scalerset_x1.append(MaxAbsScaler())

# # scalerset_x2 = [date_scaler]

# for i in range(x2.shape[1] - 1):
#     scalerset_x2.append(MinMaxScaler())

scalerset_x1 = []
scalerset_x2 = []

for i in range(x1.shape[1]):
    scalerset_x1.append(MaxAbsScaler())

for i in range(x2.shape[1]):
    scalerset_x2.append(MinMaxScaler())


ic(x1.head(), x1.tail(), x2.head(), x2.tail())

for i in range(x1.shape[1]):
    x1[i] = scalerset_x1[i].fit_transform(x1[i].to_numpy().reshape(-1, 1))

for i in range(x2.shape[1]):
    # if(i == 0):
    #     x2[i] = scalerset_x2[i].transform(x2[i].to_numpy().reshape(-1, 1))
    # else:
    #     x2[i] = scalerset_x2[i].fit_transform(x2[i].to_numpy().reshape(-1, 1))
    x2[i] = scalerset_x2[i].fit_transform(x2[i].to_numpy().reshape(-1, 1))

y = np.array([float(i) for i in y])

ic(x1.head(), x2.head(), y[:5])

'''
ic| x1.head():         0         1         2         3         4
               1  0.0000  0.071047  0.143061  0.074527  0.009200
               2  0.0001  0.071161  0.143061  0.074645  0.007053
               3  0.0002  0.071329  0.143061  0.075050  0.010426
               4  0.0003  0.071493  0.143536  0.075353  0.010120
               5  0.0004  0.071705  0.143536  0.075887  0.013186
    x2.head():        0         1         2         3         4         5         6         7    8         9         10        11    12      
               0  0.0000  0.939290  0.936937  0.912913  0.268138  0.473361  0.410256  0.157919  0.0  0.314974  0.684795  0.603175  1.00      
               1  0.0001  0.949955  0.945946  0.921922  0.199351  0.469262  0.358974  0.307987  0.0  0.716856  0.748977  0.777778  1.00      
               2  0.0002  0.948288  0.916667  0.924925  0.139779  0.489754  0.217949  0.299819  0.0  0.521727  0.750708  0.714286  1.00      
               3  0.0003  0.965384  0.959459  0.945946  0.301882  0.489754  0.397436  0.185792  0.0  0.406587  0.686526  0.714286  0.95      
               4  0.0004  0.952284  0.927928  0.942943  0.139390  0.479508  0.307692  0.240495  0.0  0.353999  0.798303  0.825397  1.00
    y[:5]: array([23., 34., 33., 43., 20.])
'''

# 전처리한 np 데이터 저장

np.save('../_save/X1_COVID_KOR_pp.npy', arr=x1)
np.save('../_save/X2_COVID_KOR_pp.npy', arr=x2)
np.save('../_save/Y_COVID_KOR_pp.npy', arr=y)

x1 = pd.DataFrame(x1)
ic(x1)
x1.columns = columns_x1
x1.to_csv('../_save/X1_COVID_KOR.csv', encoding='UTF-8')

# 열이 하나 늘어난 x1 덮어쓰기