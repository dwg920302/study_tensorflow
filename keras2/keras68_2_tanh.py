import numpy as np
import matplotlib.pyplot as plt

from icecream import ic


x = np.arange(-5, 5, 0.1)
ic(x, len(x))
'''
ic| x: array([-5.00000000e+00, -4.90000000e+00, -4.80000000e+00, -4.70000000e+00,
              -4.60000000e+00, -4.50000000e+00, -4.40000000e+00, -4.30000000e+00,
              -4.20000000e+00, -4.10000000e+00, -4.00000000e+00, -3.90000000e+00,
              -3.80000000e+00, -3.70000000e+00, -3.60000000e+00, -3.50000000e+00,
              -3.40000000e+00, -3.30000000e+00, -3.20000000e+00, -3.10000000e+00,
              -3.00000000e+00, -2.90000000e+00, -2.80000000e+00, -2.70000000e+00,
              -2.60000000e+00, -2.50000000e+00, -2.40000000e+00, -2.30000000e+00,
              -2.20000000e+00, -2.10000000e+00, -2.00000000e+00, -1.90000000e+00,
              -1.80000000e+00, -1.70000000e+00, -1.60000000e+00, -1.50000000e+00,
              -1.40000000e+00, -1.30000000e+00, -1.20000000e+00, -1.10000000e+00,
              -1.00000000e+00, -9.00000000e-01, -8.00000000e-01, -7.00000000e-01,
              -6.00000000e-01, -5.00000000e-01, -4.00000000e-01, -3.00000000e-01,
              -2.00000000e-01, -1.00000000e-01, -1.77635684e-14,  1.00000000e-01,
               2.00000000e-01,  3.00000000e-01,  4.00000000e-01,  5.00000000e-01,
               6.00000000e-01,  7.00000000e-01,  8.00000000e-01,  9.00000000e-01,
               1.00000000e+00,  1.10000000e+00,  1.20000000e+00,  1.30000000e+00,
               1.40000000e+00,  1.50000000e+00,  1.60000000e+00,  1.70000000e+00,
               1.80000000e+00,  1.90000000e+00,  2.00000000e+00,  2.10000000e+00,
               2.20000000e+00,  2.30000000e+00,  2.40000000e+00,  2.50000000e+00,
               2.60000000e+00,  2.70000000e+00,  2.80000000e+00,  2.90000000e+00,
               3.00000000e+00,  3.10000000e+00,  3.20000000e+00,  3.30000000e+00,
               3.40000000e+00,  3.50000000e+00,  3.60000000e+00,  3.70000000e+00,
               3.80000000e+00,  3.90000000e+00,  4.00000000e+00,  4.10000000e+00,
               4.20000000e+00,  4.30000000e+00,  4.40000000e+00,  4.50000000e+00,
               4.60000000e+00,  4.70000000e+00,  4.80000000e+00,  4.90000000e+00])
    len(x): 100
'''

y = np.tanh(x)

ic(y, len(y))

'''
ic| y: array([-9.99909204e-01, -9.99889103e-01, -9.99864552e-01, -9.99834566e-01,
              -9.99797942e-01, -9.99753211e-01, -9.99698579e-01, -9.99631856e-01,
              -9.99550366e-01, -9.99450844e-01, -9.99329300e-01, -9.99180866e-01,
              -9.98999598e-01, -9.98778241e-01, -9.98507942e-01, -9.98177898e-01,
              -9.97774928e-01, -9.97282960e-01, -9.96682398e-01, -9.95949359e-01,
              -9.95054754e-01, -9.93963167e-01, -9.92631520e-01, -9.91007454e-01,
              -9.89027402e-01, -9.86614298e-01, -9.83674858e-01, -9.80096396e-01,
              -9.75743130e-01, -9.70451937e-01, -9.64027580e-01, -9.56237458e-01,
              -9.46806013e-01, -9.35409071e-01, -9.21668554e-01, -9.05148254e-01,
              -8.85351648e-01, -8.61723159e-01, -8.33654607e-01, -8.00499022e-01,
              -7.61594156e-01, -7.16297870e-01, -6.64036770e-01, -6.04367777e-01,
              -5.37049567e-01, -4.62117157e-01, -3.79948962e-01, -2.91312612e-01,
              -1.97375320e-01, -9.96679946e-02, -1.77635684e-14,  9.96679946e-02,
               1.97375320e-01,  2.91312612e-01,  3.79948962e-01,  4.62117157e-01,
               5.37049567e-01,  6.04367777e-01,  6.64036770e-01,  7.16297870e-01,
               7.61594156e-01,  8.00499022e-01,  8.33654607e-01,  8.61723159e-01,
               8.85351648e-01,  9.05148254e-01,  9.21668554e-01,  9.35409071e-01,
               9.46806013e-01,  9.56237458e-01,  9.64027580e-01,  9.70451937e-01,
               9.75743130e-01,  9.80096396e-01,  9.83674858e-01,  9.86614298e-01,
               9.89027402e-01,  9.91007454e-01,  9.92631520e-01,  9.93963167e-01,
               9.95054754e-01,  9.95949359e-01,  9.96682398e-01,  9.97282960e-01,
               9.97774928e-01,  9.98177898e-01,  9.98507942e-01,  9.98778241e-01,
               9.98999598e-01,  9.99180866e-01,  9.99329300e-01,  9.99450844e-01,
               9.99550366e-01,  9.99631856e-01,  9.99698579e-01,  9.99753211e-01,
               9.99797942e-01,  9.99834566e-01,  9.99864552e-01,  9.99889103e-01])
    len(y): 100
'''

plt.plot(x, y)
plt.grid()
plt.show()