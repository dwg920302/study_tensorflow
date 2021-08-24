import tensorflow as tf

from icecream import ic


node_1 = tf.constant(3.0, tf.float32)
node_2 = tf.constant(4.0)
node_3 = tf.add(node_1, node_2)

ic(node_3)
# ic| node_3: <tf.Tensor 'Add:0' shape=() dtype=float32>

session = tf.Session()
print('node1, node2 : ', session.run([node_1, node_2]))
ic(session.run(node_3))
# node1, node2 :  [3.0, 4.0]
# ic| session.run(node_3): 7.0