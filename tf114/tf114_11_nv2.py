import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from icecream import ic
from tensorflow.python.ops.variables import global_variables_initializer


tf.set_random_seed(74)

x_1_data = [73., 93., 89., 96., 73.]     # 쿸
x_2_data = [80., 88., 91., 98., 66.]     # 0
x_3_data = [75., 93., 90., 100., 70.]    # Su
y_data = [152., 185., 180., 196., 142.]  # 결과

x_data = np.array([x_1_data, x_2_data, x_3_data]).transpose().reshape(5, 3)
y_data = np.array(y_data).transpose().reshape(5, 1)

ic(x_data)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([3, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(global_variables_initializer())

for epoch in range(2001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, '] cost : ', cost_val, '\n', hy_val)
    
session.close()