# 실습
# 1 -> 4,
# 2 -> 5, 6,
# 3 -> 6, 7, 8

import tensorflow as tf

from icecream import ic

# y = wx + b

tf.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    #, name='test)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * w + b
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())  # 초기화

for step in range(2001):
    _, loss_val, w_val, b_val= session.run([train, loss, w, b],
    feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    if step % 20 == 0:
        print(step, loss_val, w_val, b_val)
        # print(step, session.run(loss), session.run(w), session.run(b))
    
# Predict 추가

x_test_set = [[4], [5, 6], [6, 7, 8]]

pred_hypothesis = x_test * w_val + b_val

for x_test_value in x_test_set:
    pred = session.run(pred_hypothesis, feed_dict={x_test:x_test_value})

    test = optimizer.minimize(loss)

    print(pred)

'''
1900 9.549514e-07 [1.9988678] [1.0025736]
1920 8.675219e-07 [1.9989209] [1.0024527]
1940 7.8768215e-07 [1.9989716] [1.0023376]
1960 7.1569457e-07 [1.9990199] [1.0022278]
1980 6.5002195e-07 [1.9990659] [1.0021232]
2000 5.901366e-07 [1.9991096] [1.0020236]
[8.998462]
[10.997572 12.996681]
[12.996681 14.995791 16.9949  ]
'''