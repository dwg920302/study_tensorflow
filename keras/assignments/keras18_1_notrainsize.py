from sklearn.model_selection import train_test_split
import numpy as np

# HW 2 train_size의 디폴트값

# 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# train_test_split에 train_size를 안 주면 어떻게 되는지 알아보는 예제

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=42)

print(x_train.shape, x_test.shape)

# [Result] (75,) (25,)

# train과 test 둘 다 지정 안 해 줬을 때 train_size의 default 값은 0.75