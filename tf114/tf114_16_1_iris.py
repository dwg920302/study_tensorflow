import tensorflow as tf

from icecream import ic

from sklearn.datasets import load_iris

from sklearn.preprocessing import OneHotEncoder


tf.set_random_seed(93)

datasets = load_iris()

x_data = datasets.data
y_data = datasets.target.reshape(datasets.target.shape[0], 1)


encoder = OneHotEncoder(sparse=False)
y_data = encoder.fit_transform(y_data)


ic(x_data.shape, y_data.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name='weight')
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
[epoch 0 ]
ic| loss_val: 8.503992, acc_val: 0.5555556
[epoch 10 ]
ic| loss_val: 8.016085, acc_val: 0.5555556
[epoch 20 ]
ic| loss_val: 7.542909, acc_val: 0.5555556
[epoch 30 ]
ic| loss_val: 7.096689, acc_val: 0.5555556
[epoch 40 ]
ic| loss_val: 6.6947284, acc_val: 0.5555556
[epoch 50 ]
ic| loss_val: 6.3536696, acc_val: 0.5555556
[epoch 60 ]
ic| loss_val: 6.0793734, acc_val: 0.58666664
[epoch 70 ]
ic| loss_val: 5.8632236, acc_val: 0.62666667
[epoch 80 ]
ic| loss_val: 5.6891603, acc_val: 0.6888889
[epoch 90 ]
ic| loss_val: 5.5420866, acc_val: 0.72444445
[epoch 100 ]
ic| loss_val: 5.411215, acc_val: 0.74222225
...
[epoch 1900 ]
ic| loss_val: 0.55991375, acc_val: 0.8288889
[epoch 1910 ]
ic| loss_val: 0.55913806, acc_val: 0.83111113
[epoch 1920 ]
ic| loss_val: 0.5583671, acc_val: 0.83555555
[epoch 1930 ]
ic| loss_val: 0.5576008, acc_val: 0.83555555
[epoch 1940 ]
ic| loss_val: 0.55683917, acc_val: 0.8377778
[epoch 1950 ]
ic| loss_val: 0.5560821, acc_val: 0.8377778
[epoch 1960 ]
ic| loss_val: 0.5553296, acc_val: 0.8377778
[epoch 1970 ]
ic| loss_val: 0.5545817, acc_val: 0.8377778
[epoch 1980 ]
ic| loss_val: 0.55383813, acc_val: 0.8377778
[epoch 1990 ]
ic| loss_val: 0.55309904, acc_val: 0.8377778
[epoch 2000 ]
ic| loss_val: 0.55236423, acc_val: 0.8377778
'''