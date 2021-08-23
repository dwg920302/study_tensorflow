import numpy as np
import matplotlib.pyplot as plt

from icecream import ic

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# exp - exponential function (지수함수, e)

x = np.arange(1, 5)
y = softmax(x)


ic(x, len(x))
ic(y, len(y))

plt.pie(y, labels=y, shadow=False, startangle=90)
plt.show()

plt.pie(y, labels=x, shadow=False, startangle=90)
plt.show()