import tensorflow as tf

import matplotlib.pyplot as plt

from icecream import ic
from tensorflow.python.ops.variables import global_variables_initializer


tf.set_random_seed(74)

x_1_data = [73., 93., 89., 96., 73.]     # 쿸
x_2_data = [80., 88., 91., 98., 66.]     # 0
x_3_data = [75., 93., 90., 100., 70.]    # Su
y_data = [152., 185., 180., 196., 142.]  # 결과

x_1 = tf.placeholder(tf.float32)
x_2 = tf.placeholder(tf.float32)
x_3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w_1 = tf.Variable(tf.random.normal([1]), name='weight_1')
w_2 = tf.Variable(tf.random.normal([1]), name='weight_2')
w_3 = tf.Variable(tf.random.normal([1]), name='weight_3')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = x_1 * w_1 + x_2 * w_2 + x_3 * w_3 + b

cost = tf.reduce_mean(tf.square(hypothesis-y))  # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 0.01 learning rate에서는 nan 만 나와서, learning rate를 조절하는 것으로 해결함
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(global_variables_initializer())

for epoch in range(2001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train],
    feed_dict={x_1:x_1_data, x_2:x_2_data, x_3:x_3_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, '] cost : ', cost_val, '\n', hy_val)
    
session.close()