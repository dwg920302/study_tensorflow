import tensorflow as tf

from icecream import ic


tf.set_random_seed(74)

x = [1, 2, 3]
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * w + b  # y = wx + b

# 실습
# 3가지 방식으로 hypothesis 출력

session = tf.Session()
session.run(tf.global_variables_initializer())
ttt = session.run(hypothesis)
ic(ttt)
session.close()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
ttt = hypothesis.eval()
ic(ttt)
session.close()

session = tf.Session()
session.run(tf.global_variables_initializer())
ttt = hypothesis.eval(session=session)
ic(ttt)
session.close()

