#!/usr/bin/python

# -*- coding= utf-8 -*-
import tensorflow as tf
import numpy as np

a = tf.Variable(initial_value=[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]])
#a = np.asarray(a)
idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
idx2 = tf.Variable([[0, 2, 3, 0, 0], [4, 0, 2, 1, 0]], tf.int32)
init = tf.global_variables_initializer()

EMBEDDING_DIM=3
#DTYPE=tf.float32
PADID=0
mask_padding_zero_op = tf.scatter_update(a,
                                         [PADID],[[0]*EMBEDDING_DIM])
#                                         [0], [[0,0,0]])
#                                         tf.zeros([,3]))
with tf.control_dependencies([mask_padding_zero_op]):
    out1 = tf.nn.embedding_lookup(a, idx1)
    out2 = tf.nn.embedding_lookup(a, idx2)

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(out1))
    print(out1)
    print('==================')
    print(sess.run(out2))
    print(out2)
