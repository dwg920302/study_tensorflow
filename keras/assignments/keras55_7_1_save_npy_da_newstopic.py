import numpy as np
import pandas as pd

train_data = pd.read_csv('dacon/data/train_data.csv')

y_data = train_data['topic_idx']
x_data = train_data.drop('topic_idx', axis=1)

np.save('./_save/_npy/k55_x_data_dacon_newstopic.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_dacon_newstopic.npy', arr=y_data)

# 저 allow_pickle이 다른 코드와 다른 점