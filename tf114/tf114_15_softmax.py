import tensorflow as tf

import matplotlib.pyplot as plt

from icecream import ic
from tensorflow.python.ops.variables import global_variables_initializer


tf.set_random_seed(85)

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
        [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 6, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
        [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name='weight')
# w = (x.shape , y.shape)   
# (N, 4) X (4, 3) = (N, 3)
b = tf.Variable(tf.random.normal([1, 3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.binary_crossentropy(hypothesis-y))  # mse
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))   # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(1001):
    _, loss_val = session.run([optimizer, loss],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, '] loss : ', loss_val)


results = session.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
ic(results, session.run(tf.argmax(results, 1)))

# ic| results: array([[0.8708384 , 0.11846954, 0.01069199]], dtype=float32)