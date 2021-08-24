# 실습
# 위의 코드에서 lr을 수정하고, epoch를 2000이 아니라 100번 이하로 줄이기
# 결과치는 step <= 100, w = 1.9999, b = 0.9999

import tensorflow as tf

from icecream import ic

# y = wx + b

tf.set_random_seed(55)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    #, name='test)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * w + b
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1/500)

optimizer = tf.train.AdamOptimizer(learning_rate=1/10)

train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())  # 초기화

epochs = 100

for step in range(epochs):
    _, loss_val, w_val, b_val= session.run([train, loss, w, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    print(step, loss_val, w_val, b_val)
    # print(step, session.run(loss), session.run(w), session.run(b))
    
# Predict 추가
x_test = 4

pred_hypothesis = tf.norm((x_test * w_val) + b_val)

pred = session.run(pred_hypothesis)

test = optimizer.minimize(loss)

print(pred)