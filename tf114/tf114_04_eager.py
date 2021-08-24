import tensorflow as tf
from tensorflow.python.eager.context import executing_eagerly
print(tf.__version__)   # 1.14.0

print(executing_eagerly())  # True  # tf.executing_eagerly()

tf.compat.v1.disable_eager_execution()

print(executing_eagerly())  # False

# print('hello world')

hello = tf.constant('hello world')
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

session = tf.compat.v1.Session()    # v1코드
print(session.run(hello))
# b'hello world'