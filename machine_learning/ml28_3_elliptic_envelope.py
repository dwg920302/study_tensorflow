# outlier (이상치)

# 이상치 처리하기

# 1. 삭제
# 2. Nan 처리 후 -> 보간        // linear
# 3. 결측치 처리 방법과 유사하게

from icecream import ic

import matplotlib.pyplot as plt

import numpy as np

from sklearn.covariance import EllipticEnvelope


arr = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
                [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8, 9000, 10000]])

arr = arr.transpose()
ic(arr.shape)

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(arr)

results = outliers.predict(arr)

ic(results)

'''
ic| arr.shape: (10, 2)
ic| results: array([ 1,  1,  1,  1, -1,  1,  1, -1,  1,  1])
'''