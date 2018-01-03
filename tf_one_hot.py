#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
z=np.random.randint(0,10,size=[10])
y=tf.one_hot(z,10,on_value=1,off_value=None)
y1=tf.one_hot(z,10,on_value=1,off_value=None,axis=1)
with tf.Session()as sess:
    print(z)
    print(sess.run(y))
    print("axis=1 for row", sess.run(y1))
