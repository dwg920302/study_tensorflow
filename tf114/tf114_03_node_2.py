# + - * /

import tensorflow as tf

from icecream import ic


node_1 = tf.constant(2.0)
node_2 = tf.constant(3.0)
node_add = tf.add(node_1, node_2)
node_sub = tf.subtract(node_1, node_2)
node_mul = tf.multiply(node_1, node_2)
node_div = tf.divide(node_1, node_2)

ic(node_add, node_sub, node_mul, node_div)
'''
ic| node_add: <tf.Tensor 'Add:0' shape=() dtype=float32>
    node_sub: <tf.Tensor 'Sub:0' shape=() dtype=float32>
    node_mul: <tf.Tensor 'Mul:0' shape=() dtype=float32>
    node_div: <tf.Tensor 'truediv:0' shape=() dtype=float32>
'''

session = tf.Session()
print('node1, node2 : ', session.run([node_1, node_2]))
ic(session.run(node_add), session.run(node_sub), session.run(node_mul), session.run(node_div))
'''
node1, node2 :  [2.0, 3.0]
ic| session.run(node_add): 5.0
    session.run(node_sub): -1.0
    session.run(node_mul): 6.0
    session.run(node_div): 0.6666667
'''

ic(session.run([node_add, node_sub, node_mul, node_div]))
# ic| session.run([node_add, node_sub, node_mul, node_div]): [5.0, -1.0, 6.0, 0.6666667]