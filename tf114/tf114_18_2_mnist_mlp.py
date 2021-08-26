from icecream import ic

import numpy as np

import tensorflow as tf
from keras.datasets import mnist    # 114 버전에서는 keras 2.3.1 버전 이하만
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from sklearn.metrics import accuracy_score


# Stddev : A Tensor or Python value of type dtype, broadcastable with mean.
# The standard deviation of the normal distribution.

tf.set_random_seed(74)

(x_data, y_data), (x_test, y_test) = mnist.load_data()

ic(x_data.shape, y_data.shape)

# Model
node_layer_input = 28 * 28
node_layer_1 = 1024
node_layer_2 = 256
node_layer_3 = 64
node_layer_output = 10

x = tf.compat.v1.placeholder(tf.float32, shape=[None, node_layer_input])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, node_layer_output])

x_encoder = MaxAbsScaler()
x_data = x_encoder.fit_transform(x_data.reshape(-1, node_layer_input))
x_test = x_encoder.transform(x_test.reshape(-1, node_layer_input))
y_encoder = OneHotEncoder()
y_data = y_encoder.fit_transform(y_data.reshape(-1, 1)).toarray()
y_test = y_encoder.transform(y_test.reshape(-1, 1)).toarray()

# Hidden Layer(s) + hypothesis (layer)
w_1 = tf.Variable(tf.random.normal([node_layer_input, node_layer_1], stddev=0.1, name='weight_1'))
b_1 = tf.Variable(tf.random.normal([node_layer_1], stddev=0.1, name='bias_1'))

# nan!!! nan!!! nan!!! nan!!! nan!!! @#%^&*(

layer_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)           # relu

layer_1_1 = tf.nn.dropout(layer_1, keep_prob=1/2)   # Dropout

w_2 = tf.Variable(tf.random.normal([node_layer_1, node_layer_2], stddev=0.1, name='weight_2'))
b_2 = tf.Variable(tf.random.normal([node_layer_2], stddev=0.1, name='bias_2'))

layer_2 = tf.nn.relu(tf.matmul(layer_1, w_2) + b_2)     # relu

layer_2_1 = tf.nn.dropout(layer_2, keep_prob=1/2)

w_3 = tf.Variable(tf.random.normal([node_layer_2, node_layer_3], stddev=0.1, name='weight_3'))
b_3 = tf.Variable(tf.random.normal([node_layer_3], stddev=0.1, name='bias_3'))

layer_3 = tf.nn.relu(tf.matmul(layer_2_1, w_3) + b_3)     # relu

layer_3_1 = tf.nn.dropout(layer_3, keep_prob=1/2)

# Output Layer
w_out = tf.Variable(tf.random.normal([node_layer_3, node_layer_output], stddev=0.1, name='weight_out'))
b_out = tf.Variable(tf.random.normal([node_layer_output], stddev=0.1, name='bias_out'))

output_layer = tf.nn.softmax(tf.matmul(layer_3_1, w_out) + b_out)
# output_layer = tf.sigmoid(tf.matmul(layer_3_1, w_out) + b_out)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(output_layer), axis=1))   # categorical_crossentropy

# predict = tf.cast(output_layer > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

session = tf.compat.v1.Session()
session.run(tf.global_variables_initializer())

epochs = 1000 + 1

for epoch in range(epochs):
    _, loss_val = session.run([train, loss],
    feed_dict={x:x_data, y:y_data})

    # print('[ epoch', epoch, ']')
    # ic(loss_val)

    if epoch % 10 == 0:
        print('[ epoch', epoch, ']')
        ic(loss_val)

predicted = session.run(output_layer, feed_dict = {x:x_test})
y_pred = np.argmax(predicted, axis=1)
y_test = np.argmax(y_test, axis=1)
print('accuracy_score: ', accuracy_score(y_test, y_pred))

session.close()

'''
[ epoch 900 ]
ic| loss_val: 0.89331913
[ epoch 910 ]
ic| loss_val: 0.8868352
[ epoch 920 ]
ic| loss_val: 0.8772928
[ epoch 930 ]
ic| loss_val: 0.87495846
[ epoch 940 ]
ic| loss_val: 0.8682994
[ epoch 950 ]
ic| loss_val: 0.8669242
[ epoch 960 ]
ic| loss_val: 0.8609612
[ epoch 970 ]
ic| loss_val: 0.85689116
[ epoch 980 ]
ic| loss_val: 0.8575222
[ epoch 990 ]
ic| loss_val: 0.84884465
[ epoch 1000 ]
ic| loss_val: 0.8501805
accuracy_score:  0.7272
'''