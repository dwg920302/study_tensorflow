import tensorflow as tf

from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

import numpy as np

from icecream import ic
from tensorflow.python.ops.variables import global_variables_initializer

# 최종 결과값은 r2_score로 할 것

tf.set_random_seed(86)

datasets = load_boston()

x_data = datasets.data
y_data = datasets.target.reshape(datasets.target.shape[0], 1)

ic(x_data.shape, y_data.shape)

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([13, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))  # mse
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
# r2score = tf.reduce_mean(tf.cast(1 - (cost/y), dtype=tf.float32))  # Fail
# r2score = tf.reduce_mean(tf.cast(1 - (y/predict), dtype=tf.float32))  # Fail

# r2 score -> 1 - (mse/var(y))


session = tf.Session()
session.run(global_variables_initializer())

for epoch in range(101):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train],
    feed_dict={x:x_data, y:y_data})

    # if epoch % 10 == 0:
    #     print('[epoch', epoch, '] cost : ', cost_val, '\n', hy_val)

        
pre_val = session.run([predict], feed_dict={x:x_data, y:y_data})
ic(pre_val)
pre_val = np.array(pre_val)
pre_val = pre_val.reshape(pre_val.shape[1], 1)
ic(pre_val, pre_val.shape)
    
r2score = r2_score(y_data, pre_val)
ic(r2score)

session.close()

'''
ic| r2score: -5.494480321962902
'''