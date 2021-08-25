import tensorflow as tf

import matplotlib.pyplot as plt

from icecream import ic
from tensorflow.python.ops.variables import global_variables_initializer


tf.set_random_seed(74)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.binary_crossentropy(hypothesis-y))  # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 0.01 learning rate에서는 nan 만 나와서, learning rate를 조절하는 것으로 해결함
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

session = tf.Session()
session.run(global_variables_initializer())

for epoch in range(2001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, '] cost : ', cost_val, '\n', hy_val)
        

pre_val, acc_val = session.run([predict, accuracy], feed_dict={x:x_data, y:y_data})
ic(pre_val, acc_val)
    
session.close()

'''
[epoch 1900 ] cost :  0.1822803
 [[0.04564485]
 [0.17516099]
 [0.3696962 ]
 [0.75312984]
 [0.92040074]
 [0.9739667 ]]
[epoch 1910 ] cost :  0.18161575
 [[0.04532292]
 [0.17487471]
 [0.36840847]
 [0.75366783]
 [0.92079073]
 [0.97408766]]
[epoch 1920 ] cost :  0.18095599 
 [[0.04500403]
 [0.1745888 ]
 [0.3671289 ]
 [0.75420344]
 [0.921178  ]
 [0.974208  ]]
[epoch 1930 ] cost :  0.18030088
 [[0.04468812]
 [0.17430323]
 [0.36585715]
 [0.75473636]
 [0.92156243]
 [0.9743276 ]]
[epoch 1940 ] cost :  0.17965043
 [[0.04437518]
 [0.17401803]
 [0.36459336]
 [0.7552667 ]
 [0.9219442 ]
 [0.9744468 ]]
[epoch 1950 ] cost :  0.17900461
 [[0.0440652 ]
 [0.1737333 ]
 [0.36333764]
 [0.7557949 ]
 [0.9223233 ]
 [0.9745652 ]]
[epoch 1960 ] cost :  0.17836334 
 [[0.04375809]
 [0.17344886]
 [0.36208948]
 [0.75632024]
 [0.9226996 ]
 [0.9746831 ]]
[epoch 1970 ] cost :  0.17772661
 [[0.04345387]
 [0.17316487]
 [0.36084926]
 [0.7568432 ]
 [0.92307323]
 [0.9748003 ]]
[epoch 1980 ] cost :  0.17709427
 [[0.04315248]
 [0.17288125]
 [0.35961634]
 [0.75736374]
 [0.92344433]
 [0.97491676]]
[epoch 1990 ] cost :  0.17646639
 [[0.04285389]
 [0.17259806]
 [0.35839134]
 [0.75788194]
 [0.92381275]
 [0.9750328 ]]
[epoch 2000 ] cost :  0.17584282
 [[0.04255807]
 [0.17231523]
 [0.35717374]
 [0.7583977 ]
 [0.92417866]
 [0.9751482 ]]
ic| pre_val: array([[0.],
                    [0.],
                    [0.],
                    [1.],
                    [1.],
                    [1.]], dtype=float32)
    acc_val: 1.0
'''