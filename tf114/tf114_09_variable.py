import tensorflow as tf

from icecream import ic


tf.set_random_seed(74)

# y = wx + b

w = tf.Variable(tf.random_normal([1]), name='weight', dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), name='bias', dtype=tf.float32)

ic(w, b)
'''
ic| w: <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
    b: <tf.Variable 'bias:0' shape=(1,) dtype=float32_ref>
'''

session = tf.Session()
session.run(tf.global_variables_initializer())
ttt = session.run(w)
ic(ttt)
session.close()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
ttt = w.eval()
ic(ttt)
session.close()

session = tf.Session()
session.run(tf.global_variables_initializer())
ttt = w.eval(session=session)
ic(ttt)
session.close()
'''
ic| ttt: array([2.082054], dtype=float32)
ic| ttt: array([2.082054], dtype=float32)
ic| ttt: array([2.082054], dtype=float32)
'''