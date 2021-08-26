import tensorflow as tf

print(tf.__version__)   # 1.14.0

# print('hello world')

hello = tf.constant('hello world')
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

session = tf.Session()
print(session.run(hello))
# b'hello world'