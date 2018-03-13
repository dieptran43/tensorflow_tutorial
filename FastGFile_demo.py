import tensorflow as tf;    
  
image_raw_data = tf.gfile.FastGFile('/home/penglu/Desktop/11.jpg').read()  
image = tf.image.decode_jpeg(image_raw_data) #decoding  
  
print image.eval(session=tf.Session())  

'''

import tensorflow as tf;    
  
path = '/home/penglu/Desktop/11.jpg'  
file_queue = tf.train.string_input_producer([path]) #queue  
image_reader = tf.WholeFileReader()  
_, image = image_reader.read(file_queue)  
image = tf.image.decode_jpeg(image)  
  
with tf.Session() as sess:  
    coord = tf.train.Coordinator() #thread  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #start  
    print sess.run(image)  
    coord.request_stop() #stop  
    coord.join(threads) 
'''
