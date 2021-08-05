from icecream import ic
import numpy as np
import pandas as pd
import os
 

# 데이터 가져오기 (수동으로 가져와야 함)

# 코로나 데이터는 git에서 pull 해왔음, 기후 데이터는 기상청에서 직접 다운받아서 넣어야 함

# X1 만들기

# 폴더 내의 csv 파일 전부 이름을 얻고, 불러옴

path_dir = '../_data\COVID-19\csse_covid_19_data\csse_covid_19_daily_reports'
 
file_list = os.listdir(path_dir)

file_list = file_list[1:-1] # gitignore와 readme 무시

ic(len(file_list))

def retype_x1(text):
    text = text.split('.csv')[0]
    text_list = text.split('-')
    text = f'{text_list[2][2:]}{text_list[0]}{text_list[1]}'
    return text

file_list = sorted(file_list, key=lambda x : retype_x1(x))

arr = np.array([])

indexes = ['Country_Region', 'Confirmed', 'Deaths', 'Recovered']
# 가져올 것을 미리 빼냄. (파일명에서 날짜, 확진자/사망자/완치자 표기는 필수, 그 외에 취사선택) -> 변경될 수 있음

path = '../../_data/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'

# CSV 파일 순회 돌리기 (2020년 1월 22일부터))


for filename in file_list:
    ls = []
    file = pd.read_csv(path+filename)
    date = retype_x1(filename)
    # 2020년 3월 22일 이전까지는 Country_Region 컬럼이 Country/Region 컬럼으로 표기되어 있음
    '''
    try:
        file = file[indexes]
        file = file[file['Country_Region'] == 'Korea, South']
    except KeyError:
        file = file['Country/Region', 'Confirmed', 'Deaths', 'Recovered']
        file = file[file['Country/Region'] == 'South Korea']
    '''

    indexes = ['Country_Region', 'Confirmed', 'Deaths', 'Recovered']

    if int(date) >= 200331:     # 2020년 4월 1일부터, 단 4월 1일의 확진자 수를 알려면 3월 31일 데이터가 필요함
        file = file[indexes]
        file = file[file['Country_Region'] == 'Korea, South']
        file = file.drop('Country_Region', axis=1)
        ic(file)
        ls.append(retype_x1(filename))
        ls = np.array(ls)
        ls = np.append(ls, file)
        ic(ls.shape)
        arr = np.append(arr, np.array(ls))
        ic(arr.shape)
    else:
        # file = file['Country/Region', 'Confirmed', 'Deaths', 'Recovered']
        # file = file[file['Country/Region'] == 'South Korea']
        pass

print('-'*100)

arr = arr.reshape(int(arr.shape[0]/4), 4) 
ic(arr, arr.shape)  # (36X, 4)

# 5~600개의 CSV 파일에서 일단 South Korea만 가져와서, np.array에 넣음

# 빼온 걸 array 형태로 저장하여 CSV나 npy를 따로 만듦.

columns = ['Date', 'Confirmed_Total', 'Deaths_Total', 'Recovered_Total']

np.save('../_save/X1_COVID_KOR.npy', arr=arr)
arr = pd.DataFrame(arr)
ic(arr)
arr.columns = columns
arr.to_csv('../_save/X1_COVID_KOR.csv', encoding='UTF-8')


# X2 만들기

# 기상청 날씨 데이터 (2020년 8월 1일부터)

# 인코딩이 되어있지 않아서 인코딩한 CSV를 다시 만듦

filename = '../_data/COVID-19/OBS/OBS_ASOS_DD_20210804173000.csv'

file = pd.read_csv(filename, encoding='euc_kr', low_memory=False)

ic(file.shape, file.head(), file.tail())

file.to_csv('../_data/COVID-19/OBS/OBS_encoded_20210804173000.csv', encoding='UTF-8')

filename = '../_data/COVID-19/OBS/OBS_encoded_20210804173000.csv'

# file = pd.read_csv(filename, encoding='euc_kr', low_memory=False)

file = pd.read_csv(filename)

ic(file.shape, file.head(), file.tail())

def retype_x2(text):
    text_list = text.split('-')
    text = f'{text_list[0][2:]}{text_list[1]}{text_list[2]}'
    return text

file['일시'] = file['일시'].map(lambda x: retype_x2(x))

file = file.fillna('0.0')

ic(file.shape, file.head(), file.tail())

# 컬럼수가 62개인데 이 중에서 필요한 것만 빼옴

columns = ['일시', '평균기온(°C)', '최고기온(°C)', '일강수량(mm)', '최소 상대습도(%)']

file = file[columns]

ic(file.shape, file.head(), file.tail()) # (34769, 5)

# (X1과 똑같이 365열로 줄여야 함, 인풋은 늘어나도 줄어도 상관없음)

dates = np.unique(file['일시'])

ic(dates)

arr = np.array([])

for date in dates:
    tmp = file.loc[file['일시'] == date]
    ic(tmp.shape)

    ls = np.array([])

    ls = np.append(ls, date)

    for col in columns[1:]:
        tmp_avg = sum(tmp[col].map(lambda x: float(x))) / tmp.shape[0]
        tmp_min = min(tmp[col].map(lambda x: float(x)))
        tmp_max = max(tmp[col].map(lambda x: float(x)))
        # ic(tmp_avg, tmp_min, tmp_max)
        
        # 5,6,7번째 행을 최고기온에서 최고 온도차로 바꿈 (5 = 5-2, 6 = 6-3, 7 = 7-4)

        if(col == '최고기온(°C)'):
            tmp_avg -= float(ls[1])
            tmp_min -= float(ls[2])
            tmp_max -= float(ls[3])

        ls = np.append(ls, (tmp_avg, tmp_min, tmp_max))

    arr = np.append(arr, np.array(ls))

ic(arr, arr.shape) # arr.shape: (4758,)

arr = arr.reshape(int(arr.shape[0]/13), 13) #  arr.shape: (366, 13)

ic(arr, arr.shape) # arr.shape: (365, 13)

columns = ['Date',
            'avg_temp_avg', 'avg_temp_min', 'avg_temp_max', 
            'tmp_diff_avg', 'tmp_diff_min', 'tmp_diff_max',
            'rainfall_avg', 'rainfall_min', 'rainfall_max',
            'humidity_avg', 'humidity_min', 'humidity_max']

np.save('../_save/X2_COVID_KOR.npy', arr=arr)
arr = pd.DataFrame(arr)
ic(arr)
arr.columns = columns
arr.to_csv('../_save/X2_COVID_KOR.csv', encoding='UTF-8')