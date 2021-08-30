from icecream import ic

import tensorflow as tf
import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.models import Sequential


tf.set_random_seed(99)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])


# Modeel

w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])     # 알아서 초기값을 넣어줌, shape와 이름이 반드시 들어가야 함
l1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
#   tf1은 padding을 대문자로 입력해야 함;
# w2 = tf.Variable(tf.random.normal([3, 3, 1, 32]), dtype=tf.float32)
# w3 = tf.Variable([1], dtype=tf.float32)

print(w1)
print(l1)

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1)))

# session = tf.Session()
# session.run(tf.global_variables_initializer())
# print(np.min(session.run(w1)))
# print(np.max(session.run(w1)))
# print(np.mean(session.run(w1)))
# print(np.median(session.run(w1)))
# print(session.run(w1))
# print(w1)