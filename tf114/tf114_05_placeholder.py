import tensorflow as tf

from icecream import ic

node_1 = tf.constant(3.0, tf.float32)
node_2 = tf.constant(4.0)
node_3 = tf.add(node_1, node_2)

ic(node_3)
# ic| node_3: <tf.Tensor 'Add:0' shape=() dtype=float32>

session = tf.compat.v1.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

ic(session.run(adder_node, feed_dict={a:3, b:4.5}))
ic(session.run(adder_node, feed_dict={a:[1, 3], b:[3, 4]}))

# ic| session.run(adder_node, feed_dict={a:3, b:4.5}): 7.5
# ic| session.run(adder_node, feed_dict={a:[1, 3], b:[3, 4]}): array([4., 7.], dtype=float32)

add_and_triple = adder_node * 3

ic(session.run(add_and_triple, feed_dict={a:[4], b:[2]}))

# ic| session.run(add_and_triple, feed_dict={a:[4], b:[2]}): array([18.], dtype=float32)