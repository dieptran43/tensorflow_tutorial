import tensorflow as tf

w_1 = tf.Variable(3,name="w_1")
w_2 = tf.Variable(1,name="w_1")
print w_1.name
print w_2.name
#output
#w_1:0
#w_1_1:0

#w_1 = tf.get_variable(name="w_1",initializer=1)
#w_2 = tf.get_variable(name="w_1",initializer=2)
#Error
#ValueError: Variable w_1 already exists, disallowed. Did
#you mean to set reuse=True in VarScope?

#When we need share variable, we can only use get_variable
with tf.variable_scope("scope1"):
    w1 = tf.get_variable("w1", shape=[])
    w2 = tf.Variable(0.0, name="w2")
with tf.variable_scope("scope1", reuse=True):
    w1_p = tf.get_variable("w1", shape=[])
    w2_p = tf.Variable(1.0, name="w2")

print(w1 is w1_p, w2 is w2_p)
#output
#True  False
