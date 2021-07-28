import numpy as np
from icecream import ic

x_data = np.load('../_save/_npy/k55_x_data_wine.npy')
y_data = np.load('../_save/_npy/k55_y_data_wine.npy')

ic(x_data.shape, y_data.shape)
ic(x_data, y_data)

'''
ic| x_data: array([[ 7.  ,  0.27,  0.36, ...,  3.  ,  0.45,  8.8 ],
                   [ 6.3 ,  0.3 ,  0.34, ...,  3.3 ,  0.49,  9.5 ],
                   [ 8.1 ,  0.28,  0.4 , ...,  3.26,  0.44, 10.1 ],
                   ...,
                   [ 6.5 ,  0.24,  0.19, ...,  2.99,  0.46,  9.4 ],
                   [ 5.5 ,  0.29,  0.3 , ...,  3.34,  0.38, 12.8 ],
                   [ 6.  ,  0.21,  0.38, ...,  3.26,  0.32, 11.8 ]])
    y_data: array([6, 6, 6, ..., 6, 7, 6], dtype=int64)
'''