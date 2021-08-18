# outlier (이상치)

# 이상치 처리하기

# 1. 삭제
# 2. Nan 처리 후 -> 보간        // linear
# 3. 결측치 처리 방법과 유사하게

from icecream import ic

import matplotlib.pyplot as plt

import numpy as np

arr = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100], [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8, 9000, 10000]])

def outliers(data_out):
        quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
        print("1사분위 : ", quartile_1)
        print("q2 : ", q2)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out>upper_bound) | (data_out<lower_bound))

def outliers_all(data_out):
        ls = ['', ]
        ls.remove('')
        for line in data_out:
                ic(line)

                quartile_1, q2, quartile_3 = np.percentile(line, [25, 50, 75])
                print("1사분위 : ", quartile_1)
                print("q2 : ", q2)
                print("3사분위 : ", quartile_3)
                iqr = quartile_3 - quartile_1
                lower_bound = quartile_1 - (iqr * 1.5)
                upper_bound = quartile_3 + (iqr * 1.5)

                ls.append(np.where((line>upper_bound) | (line<lower_bound)))

                ic(ls)
        return ls              

# outliers_loc = outliers(arr)

# ic(outliers_loc)

outliers_loc_all = outliers_all(arr)

ic(outliers_loc_all)


# 시각화 (BoxPlot으로)

plt.boxplot(arr)

plt.show()