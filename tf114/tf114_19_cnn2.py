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
l1 = tf.nn.relu(l1)
# Padding을 VALID로 할 경우
# <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>
# Tensor("Conv2D:0", shape=(?, 26, 26, 32), dtype=float32)
l1_maxpool = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# ksize = kernel_size와 비슷한 역할로, 양쪽 1은 들러리(차원 맞춰주는 역할)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
                padding='same', input_shape=(28, 28, 1), activation='relu'))

print(w1)
print(l1)
print(l1_maxpool)   # (14, 14, 32). 마지막 열은 개수 변화가 없음

w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])    # kernel_size, input, output
l2 = tf.nn.conv2d(l1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')
l2 = tf.nn.elu(l2)
l2_maxpool = tf.nn.max_pool(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(l2)
print(l2_maxpool)
# Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])
l3 = tf.nn.conv2d(l2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
l3 = tf.nn.selu(l3)
l3_maxpool = tf.nn.max_pool(l3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(l3)
print(l3_maxpool)
# Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

w4 = tf.get_variable('w4', shape=[2, 2, 128, 64])
l4 = tf.nn.conv2d(l3_maxpool, w4, strides=[1, 1, 1, 1], padding='VALID')
l4 = tf.nn.relu(l4)
l4_maxpool = tf.nn.max_pool(l4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(l4)
print(l4_maxpool)
# Tensor("Relu_1:0", shape=(?, 3, 3, 64), dtype=float32)
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
l_flat = tf.reshape(l4_maxpool, [-1, 2*2*64])   # tf 1에서는 따로 메소드가 구현되어있진 않음
print(l_flat)   # Tensor("Reshape:0", shape=(?, 256), dtype=float32)

w5 = tf.get_variable('w5', shape=[2*2*64, 64])    # last
b5 = tf.Variable(tf.random.normal([64]), name='bias5')
l5 = tf.matmul(l_flat, w5) + b5
l5 = tf.nn.selu(l5)
l5 = tf.nn.dropout(l5, keep_prob=0.2)
print(l5)
# Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

w6 = tf.get_variable('w6', shape=[64, 32])    # last
b6 = tf.Variable(tf.random.normal([32]), name='bias6')
l6 = tf.matmul(l5, w6) + b6
l6 = tf.nn.selu(l6)
l6 = tf.nn.dropout(l6, keep_prob=0.2)
print(l6)
# Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

# Softmax
w7 = tf.get_variable('w7', shape=[32, 10])    # last
b7 = tf.Variable(tf.random.normal([10]), name='bias7')
l7 = tf.matmul(l6, w7) + b7
hypothesis = tf.nn.softmax(l7)
print(hypothesis)