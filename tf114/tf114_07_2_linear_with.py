import tensorflow as tf

from icecream import ic

# y = wx + b

tf.set_random_seed(66)

x_train = [1, 2, 3]
y_train = [3, 5, 7]

w = tf.Variable(2, dtype=tf.float32)    #, name='test)
b = tf.Variable(3, dtype=tf.float32)

hypothesis = x_train * w + b
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())  # 초기화

for step in range(2001):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(loss), session.run(w), session.run(b))