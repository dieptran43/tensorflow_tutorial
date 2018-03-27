import tensorflow as tf
  sess = tf.Session()
  red = tf.constant([255, 0, 0])
  file_names = ['./images/chapter-05-object-recognition-and-classification/working-with-images/test-input-image.jpg']
  filename_queue = tf.train.string_input_producer(file_names)
  image_reader = tf.WholeFileReader()
  _, image_file = image_reader.read(filename_queue)
  image = tf.image.decode_jpeg(image_file)
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  print sess.run(image)
  filename_queue.close(cancel_pending_enqueues=True)
  coord.request_stop()
  coord.join(threads)
  print "------------------------------------------------------"
  image_label = b'\x01'
  image_loaded = sess.run(image)
  image_bytes = image_loaded.tobytes()
  image_height, image_width, image_channels = image_loaded.shape
  writer = tf.python_io.TFRecordWriter("./output/training-image.tfrecord")
  example = tf.train.Example(features=tf.train.Features(feature={
          'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
          'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
      }))
  print example
  writer.write(example.SerializeToString())
  writer.close()
  print "------------------------------------------------------"
  tf_record_filename_queue = tf.train.string_input_producer(["./output/training-image.tfrecord"])
  tf_record_reader = tf.TFRecordReader()
  _, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)
  tf_record_features = tf.parse_single_example(
  tf_record_serialized,
  features={
      'label': tf.FixedLenFeature([], tf.string),
      'image': tf.FixedLenFeature([], tf.string),
      })
  tf_record_image = tf.decode_raw(
      tf_record_features['image'], tf.uint8)
  tf_record_image = tf.reshape(
      tf_record_image,
      [image_height, image_width, image_channels])
  print tf_record_image
  tf_record_label = tf.cast(tf_record_features['label'], tf.string)
  print tf_record_label
  print "------------------------------------------------------"
  sess.close()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  print sess.run(tf.equal(image, tf_record_image))
  sess.run(tf_record_label)
  coord.request_stop()
  coord.join(threads)
  print "------------------------------------------------------"
  print sess.run(tf.image.central_crop(image, 0.1))
  real_image = sess.run(image)
  bounding_crop = tf.image.crop_to_bounding_box(
      real_image, offset_height=0, offset_width=0, target_height=2, target_width=1)
  print sess.run(bounding_crop)
  print "------------------------------------------------------"
  real_image = sess.run(image)
  pad = tf.image.pad_to_bounding_box(
      real_image, offset_height=0, offset_width=0, target_height=4, target_width=4)
  print sess.run(pad)
  print "------------------------------------------------------"
  crop_or_pad = tf.image.resize_image_with_crop_or_pad(
      real_image, target_height=2, target_width=5)
  print sess.run(crop_or_pad)
  print "------------------------------------------------------"
  sess.close()
  sess = tf.Session()
  top_left_pixels = tf.slice(image, [0, 0, 0], [2, 2, 3])
  flip_horizon = tf.image.flip_left_right(top_left_pixels)
  flip_vertical = tf.image.flip_up_down(flip_horizon)
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  print sess.run([top_left_pixels, flip_vertical])
  print "------------------------------------------------------"
  top_left_pixels = tf.slice(image, [0, 0, 0], [2, 2, 3])
  random_flip_horizon = tf.image.random_flip_left_right(top_left_pixels)
  random_flip_vertical = tf.image.random_flip_up_down(random_flip_horizon)
  print sess.run(random_flip_vertical)
  print "------------------------------------------------------"
  example_red_pixel = tf.constant([254., 2., 15.])
  adjust_brightness = tf.image.adjust_brightness(example_red_pixel, 0.2)
  print sess.run(adjust_brightness)
  print "------------------------------------------------------"
  adjust_contrast = tf.image.adjust_contrast(image, -.5)
  print sess.run(tf.slice(adjust_contrast, [1, 0, 0], [1, 3, 3]))
  print "------------------------------------------------------"
  adjust_hue = tf.image.adjust_hue(image, 0.7)
  print sess.run(tf.slice(adjust_hue, [1, 0, 0], [1, 3, 3]))
  print "------------------------------------------------------"
  adjust_saturation = tf.image.adjust_saturation(image, 0.4)
  print sess.run(tf.slice(adjust_saturation, [1, 0, 0], [1, 3, 3]))
  print "------------------------------------------------------"
  gray = tf.image.rgb_to_grayscale(image)
  print sess.run(tf.slice(gray, [0, 0, 0], [1, 3, 1]))
  print "------------------------------------------------------"
  hsv = tf.image.rgb_to_hsv(tf.image.convert_image_dtype(image, tf.float32))
  print sess.run(tf.slice(hsv, [0, 0, 0], [3, 3, 3]))
  print "------------------------------------------------------"
  rgb_hsv = tf.image.hsv_to_rgb(hsv)
  rgb_grayscale = tf.image.grayscale_to_rgb(gray)
  print rgb_hsv, rgb_grayscale
  print "------------------------------------------------------"
