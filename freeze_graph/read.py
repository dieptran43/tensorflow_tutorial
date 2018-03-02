import tensorflow as tf

new_input = tf.placeholder(tf.float32, shape=())

with tf.Session() as sess:
    with open('./graph.pb', 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        output = tf.import_graph_def(graph_def, input_map={'input:0':new_input}, return_elements=['out:0'], name='a') 
        print(sess.run(output, feed_dict={new_input:4}))
