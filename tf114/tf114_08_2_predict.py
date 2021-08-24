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
    _, loss_val, w_val, b_val= session.run([train, loss, w, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    if step % 20 == 0:
        print(step, loss_val, w_val, b_val)
        # print(step, session.run(loss), session.run(w), session.run(b))
    
# Predict 추가
# 현재 단일은 구현되었는데, 복수 predict를 하면(리스트로 넣으면) 이상한 값이 나옴
x_test = 4

pred_hypothesis = tf.norm((x_test * w_val) + b_val)

pred = session.run(pred_hypothesis)

test = optimizer.minimize(loss)

print(pred)