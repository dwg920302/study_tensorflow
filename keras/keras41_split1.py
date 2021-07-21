import numpy as np

a = np.array(range(1, 11))
size = 5

# 시계열 데이터의 X와 Y 분리시키는 함수 분석하기

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

print(dataset)



# x와 y 분리

x = dataset[:, :4]
y = dataset[:, 4]

print("x : ", x, " / ")
print("y : ", y)

'''
x :  [[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]  /
y :  [ 5  6  7  8  9 10]
'''