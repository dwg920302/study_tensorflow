import tensorflow as tf

from icecream import ic

from sklearn.datasets import load_wine

from sklearn.preprocessing import OneHotEncoder


tf.set_random_seed(81)

datasets = load_wine()

x_data = datasets.data
y_data = datasets.target.reshape(datasets.target.shape[0], 1)


encoder = OneHotEncoder(sparse=False)
y_data = encoder.fit_transform(y_data)


ic(x_data.shape, y_data.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([13, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))   # categorical_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(2001):
    _, loss_val, acc_val = session.run([optimizer, loss, accuracy],
    feed_dict={x:x_data, y:y_data})

    if epoch % 10 == 0:
        print('[epoch', epoch, ']')
        ic(loss_val, acc_val)


'''
[epoch 900 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 910 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 920 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 930 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 940 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 950 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 960 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 970 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 980 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 990 ]
ic| loss_val: nan, acc_val: 0.6666667
[epoch 1000 ]
ic| loss_val: nan, acc_val: 0.6666667
'''