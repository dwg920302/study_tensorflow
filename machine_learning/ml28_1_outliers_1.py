# outlier (이상치)

# 이상치 처리하기

# 1. 삭제
# 2. Nan 처리 후 -> 보간        // linear
# 3. 결측치 처리 방법과 유사하게
# 4. 스케일러
# 5. 모델링 : tree계열, DT, RF, XG, LGBM

from icecream import ic

import matplotlib.pyplot as plt

import numpy as np

# arr = np.array([1, 2, -10000, 4, 5, 6, 7, 8, 90, 100, 5000])

arr = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])

def outliers(data_out):
        quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
        print("1사분위 : ", quartile_1)
        print("q2 : ", q2)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(arr)

ic(outliers_loc)
# outlier의 개수를 count하는 기능을 추가할 것

# 시각화 (BoxPlot으로)

plt.boxplot(arr)

plt.show()

