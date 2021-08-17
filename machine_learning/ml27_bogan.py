# very very very important

# [1, np.nan, np.nan, 8, 10]

# 결측치 처리

# 1. 결측치가 있는 행을 삭제함
# 2. 특정 값을 일괄적으로 넣어줌 (ex. 0, 1(앞의 값), 8(뒤의 값), 4.5(중위값))
# 3. 보간
# 4. 모델링 - predict(결측지들을 전부 빼고 훈련 -> 결측치를 predict)
# 부스트는 결측치에 대해 비교적 자유로움

from icecream import ic
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime

date_strs=['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(date_strs)
ic(dates)

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
ic(ts)

ts_intp_linear = ts.interpolate()
ic(ts_intp_linear)

'''
ic| ts: 2021-08-13     1.0
        2021-08-14     NaN
        2021-08-15     NaN
        2021-08-16     8.0
        2021-08-17    10.0
        dtype: float64
ic| ts_intp_linear: 2021-08-13     1.000000
                    2021-08-14     3.333333
                    2021-08-15     5.666667
                    2021-08-16     8.000000
                    2021-08-17    10.000000
'''