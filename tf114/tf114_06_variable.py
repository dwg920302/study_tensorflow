import tensorflow as tf

from icecream import ic


session = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

# Tensorflow의 변수는 반드시 초기화를 하고 나서 사용해야 함.
init = tf.global_variables_initializer()

session.run(init)   # 초기화가 실행되는 시점
print(session.run(x))   # [2.]