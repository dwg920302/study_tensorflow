import tensorflow as tf

import matplotlib.pyplot as plt

from icecream import ic


tf.set_random_seed(74)

x = [1., 2., 3.]
y = [3., 6., 9.]

w = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis - y))    # loss

w_history = []
cost_history = []

with tf.compat.v1.Session() as session:
    for i in range(-30, 50):
        current_w = i
        current_cost = session.run(cost, feed_dict={w:current_w})

        w_history.append(current_w)
        cost_history.append(current_cost)

print("="*50)
ic(w_history)
print("="*50)
ic(cost_history)
print("="*50)

plt.plot(w_history, cost_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()