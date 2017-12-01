# coding: utf-8

import tensorflow as tf
from tensorflow.contrib import slim
import matplotlib.image as mpimg
import time
import sys
sys.path.append('../')
from nets import myssd_vgg_300 as ssd_vgg_300
from preprocessing import tf_image
import numpy as np
import  scipy


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

net_shape = (300, 300)
data_format = 'NHWC'

image_4d_input = tf.placeholder(dtype=tf.uint8, shape=(None, 300, 300, 3))
image_4d_input = tf.to_float(image_4d_input)
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    x = ssd_net.net(image_4d_input, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'

isess = tf.Session(config=config)
init = tf.global_variables_initializer()
ssd_anchors = ssd_net.anchors(net_shape)

# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
writer = tf.summary.FileWriter("logs/", isess.graph)

isess.run(init)
# saver = tf.train.Saver()
# saver.restore(isess, ckpt_filename)


image = scipy.resize(mpimg.imread('dog.jpg'),(300,300,3))
image2 = scipy.resize(mpimg.imread('dog.jpg'),(300,300,3))
images = [image,image2]



test_batch = [i for i in range(1,3)]
for i,size in enumerate(test_batch):
    for j in range(10):
        imgs_input = [images[j%2] for ddd in range(size)]
        start = time.time()
        all_box = isess.run([x], feed_dict={image_4d_input: imgs_input})
        consume_time = (time.time() - start) * 1000
        print j,size,consume_time,consume_time/size

