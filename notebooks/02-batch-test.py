# coding: utf-8

import tensorflow as tf


slim = tf.contrib.slim



import matplotlib.image as mpimg




import sys

sys.path.append('../')




from nets import myssd_vgg_300 as  ssd_vgg_300
from preprocessing import ssd_vgg_preprocessing



gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

isess = tf.Session()

net_shape = (300, 300)
data_format = 'NHWC'

img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))




image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None,
                                                                                        net_shape, data_format,
                                                                                        resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

batch_size = 5
image_pre = tf.expand_dims(image_pre, 0)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations  = ssd_net.net(image_4d, is_training=False, reuse=reuse)


ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'

init = tf.global_variables_initializer()

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(init)
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)


# ## Post-processing pipeline
#
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
#
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[11]:


img = mpimg.imread('dog.jpg')

import time
start = time.time()
all_box = isess.run([predictions, localisations], feed_dict={img_input: img })
consume_time = (time.time() - start) * 1000
print consume_time,consume_time/batch_size

