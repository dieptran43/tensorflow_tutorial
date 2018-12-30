import tf
import os

#specify the gpu device id 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#specify the fraction of all the gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   

#specify the allow_growth option
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
