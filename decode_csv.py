import tensorflow as tf
import numpy as np

path = '/home/shr/software/machine_learning/kaggle/train.csv'
filename_queue = tf.train.string_input_producer([path])
num_epoch = 1
learning_rate = 0.01

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]
    , [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]
    , [1], [1], [1], [1], [1],[1], [1], [1]]

col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, \
col15, co16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, \
co27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, \
col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49 = \
    tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14,
                     col15, co16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26,
                     co27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38,
                     col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49])

label = col2

features_batch, label_batch = tf.train.shuffle_batch(
    [features, label], batch_size=32, capacity=640,
    min_after_dequeue=320)

print ("Shape of feature batch is : {}".format(features_batch.shape))
print ("Shape of label batch is : {}".format(label_batch.shape))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for _ in range(10):  # generate 10 batches
        features, labels = sess.run([features_batch, label_batch])
        print(features)
    coord.request_stop()
    coord.join(threads)
