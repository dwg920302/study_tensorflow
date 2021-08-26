from icecream import ic

import numpy as np

import tensorflow as tf
from keras.datasets import mnist    # 114 버전에서는 keras 2.3.1 버전 이하만
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from sklearn.metrics import accuracy_score


tf.set_random_seed(74)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, y_train.shape)

# Model
node_layer_input = 28 * 28
node_layer_1 = 100
node_layer_output = 10

x = tf.compat.v1.placeholder(tf.float32, shape=[None, node_layer_input])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, node_layer_output])

x_encoder = MaxAbsScaler()
x_train = x_encoder.fit_transform(x_train.reshape(-1, node_layer_input))
x_test = x_encoder.transform(x_test.reshape(-1, node_layer_input))
y_encoder = OneHotEncoder()
y_train = y_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = y_encoder.transform(y_test.reshape(-1, 1)).toarray()

# Hidden Layer(s) + hypothesis (layer)
w_1 = tf.Variable(tf.random.normal([node_layer_input, node_layer_1], stddev=0.1, name='weight_1'))
b_1 = tf.Variable(tf.random.normal([node_layer_1], stddev=0.1, name='bias_1'))

# nan!!! nan!!! nan!!! nan!!! nan!!! @#%^&*(!@%$!#^ @#$@!#
# stddev를 넣으니까 됨

layer_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)           # relu

# Output Layer
w_out = tf.Variable(tf.random.normal([node_layer_1, node_layer_output], stddev=0.1, name='weight_out'))
b_out = tf.Variable(tf.random.normal([node_layer_output], stddev=0.1, name='bias_out'))

output = tf.nn.softmax(tf.matmul(layer_1, w_out) + b_out)

loss = tf.reduce_mean(-tf.reduce_sum(y* tf.math.log(output), axis=1)) # categorical_crossentropy

# predict = tf.cast(output_layer > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

# train = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

session = tf.compat.v1.Session()
session.run(tf.global_variables_initializer())

epochs = 500 + 1

for epoch in range(epochs):
    _, loss_val = session.run([train, loss], feed_dict={x:x_train, y:y_train})

    print('[ epoch', epoch, ']')
    ic(loss_val)

    # if epoch % 10 == 0:
    #     print('[epoch', epoch, ']')
    #     ic(loss_val)

# results = session.run([output_layer], feed_dict = {x:x_test})
# ic(results)

predicted = session.run(output, feed_dict = {x:x_test})
y_pred = np.argmax(predicted, axis=1)
y_test = np.argmax(y_test, axis=1)
print('accuracy_score: ', accuracy_score(y_test, y_pred))

session.close()

'''
[ epoch 490 ]
ic| loss_val: 0.75430715
[ epoch 491 ]
ic| loss_val: 0.7533952
[ epoch 492 ]
ic| loss_val: 0.7524868
[ epoch 493 ]
ic| loss_val: 0.7515818
[ epoch 494 ]
ic| loss_val: 0.75068045
[ epoch 495 ]
ic| loss_val: 0.74978256
[ epoch 496 ]
ic| loss_val: 0.748888
[ epoch 497 ]
ic| loss_val: 0.74799687
[ epoch 498 ]
ic| loss_val: 0.74710906
[ epoch 499 ]
ic| loss_val: 0.74622446
[ epoch 500 ]
ic| loss_val: 0.7453431
accuracy_score:  0.8173
PS D:\study> 
'''