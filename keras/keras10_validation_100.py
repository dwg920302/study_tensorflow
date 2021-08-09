# keras-010 #3-A [Validation Percentage]

# train_test_split을 2번 했을 때 어떻게 나뉘는지 보기 위한 예제

from icecream import ic

import numpy as np

from sklearn.model_selection import train_test_split


x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=42)

ic(x_train.shape, x_test.shape, x_valid.shape) # (64,) (20,) (16,)

# 여기서 60/20/20으로 나누려면 2번째 나눌 때 test_size를 0.25로 줘야 함.