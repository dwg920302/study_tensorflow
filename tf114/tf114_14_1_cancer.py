import tensorflow as tf

from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

from icecream import ic
from tensorflow.python.ops.variables import global_variables_initializer

# 최종 결과값은 r2_score로 할 것

tf.set_random_seed(55)

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target.reshape(datasets.target.shape[0], 1)

ic(x_data.shape, y_data.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([30, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y))  # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
# r2score = tf.reduce_mean(1 - (cost/y))

session = tf.Session()
session.run(global_variables_initializer())

for epoch in range(1001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, '] cost : ', cost_val, '\n', hy_val)
        

pre_val, acc_val = session.run([predict, accuracy], feed_dict={x:x_data, y:y_data})
# pre_val, r2_val = session.run([predict, r2score], feed_dict={x:x_data, y:y_data})
ic(pre_val, acc_val)
# ic(pre_val, r2_val)
    
session.close()

#acc_val: 0.40070298