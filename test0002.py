import numpy as np

x = np.array(range(100))
# y = np.array(range(1, 101))

np.random.shuffle(x)
y = x
map(lambda a : a+1, y)

# y.sort(lambda w : x[w])

print('x = ', x)
print('y = ', y)