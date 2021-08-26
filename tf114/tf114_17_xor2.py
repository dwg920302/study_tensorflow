from icecream import ic

import tensorflow as tf


tf.set_random_seed(66)

# Data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

# Model
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Hidden Layer(s) + hypothesis (layer)
w_1 = tf.Variable(tf.random.normal([2, 16]), name='weight_1')    # Node 3개
b_1 = tf.Variable(tf.random.normal([16]), name='bias_1')         # 행렬 연산을 함

layer_1 = tf.matmul(x, w_1) + b_1    # linear

w_2 = tf.Variable(tf.random.normal([16, 16]), name='weight_2')    # Node 4개
b_2 = tf.Variable(tf.random.normal([16]), name='bias_2')         # 앞 노드가 3개였으므로 3개로 받고, 다음 노드 수(4개)를 정함

layer_2 = tf.matmul(layer_1, w_2) + b_2    # nan jungmal sigmoid

# Output Layer
w_out = tf.Variable(tf.random.normal([16, 1]), name='weight_out')    # 맨 마지막 node 수가 output이 됨 (1)
b_out = tf.Variable(tf.random.normal([1]), name='bias_out')

output_layer = tf.sigmoid(tf.matmul(layer_2, w_out) + b_out)    # nan jungmal sigmoid

cost = -tf.reduce_mean(y * tf.log(output_layer) + (1 - y) * tf.log(1 - output_layer))       # binary

predict = tf.cast(output_layer > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

session = tf.compat.v1.Session()
session.run(tf.global_variables_initializer())

for epoch in range(5001):
    _, loss_val, acc_val = session.run([optimizer, cost, accuracy],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, ']')
        ic(loss_val, acc_val)
        if acc_val == 1:    # EarlyStopping 비슷한 느낌
            break

results = session.run([output_layer, predict, accuracy], feed_dict = {x:x_data,y:y_data})
ic(results[0])
print('Prediction : ', results[1], '  Accuracy : ', results[2])


# Acccuracy 1 만들기