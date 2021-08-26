from icecream import ic

import tensorflow as tf


tf.set_random_seed(66)

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))       # binary

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

session = tf.compat.v1.Session()
session.run(tf.global_variables_initializer())

for epoch in range(1001):
    _, loss_val, acc_val = session.run([optimizer, cost, accuracy],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, ']')
        ic(loss_val, acc_val)