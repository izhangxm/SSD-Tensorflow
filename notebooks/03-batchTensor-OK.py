# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import sys
sys.path.append('../')

from nets import ssd_vgg_300 as ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)





data_format = 'NHWC'
batch_size = 2
net_shape = (300, 300)





img_input = tf.placeholder(tf.uint8, shape=(None, 300, 300, 3))


img_list = []
for i in range(batch_size):
    with tf.name_scope('prepare_image_'+ str(batch_size)):
        image_pre, _, _, _ = ssd_vgg_preprocessing.preprocess_for_eval(img_input[i], None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    img_list.append(image_pre)
img_4d = tf.stack(img_list,axis=0)


ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _  = ssd_net.net(img_4d, is_training=False, reuse=None)


isess = tf.Session(config=config)
writer = tf.summary.FileWriter("logs/", isess.graph)
init = tf.global_variables_initializer()
ssd_anchors = ssd_net.anchors(net_shape)
isess.run(init)


# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)


"""
处理成batch形式，方便批量操作

@:param rpredictions   box_layer_n x batch x H x W x Boxn x Clsn
@:param rlocalisations box_layer_n x batch x H x W x Boxn x 4
@:return bat_rpredictions(batch x box_layer_n x H x W x Boxn x Clsn),bat_rlocalisations(batch x box_layer_n x H x W x Boxn x 4)
"""
def re_construct_result(rpredictions, rlocalisations):

    # box_layer_depth
    len_rpredictions = len(rpredictions)
    # batch_size
    len_bat = rpredictions[0].shape[0]

    # bat and init-setup:1/2
    bat_rpredictions = [i for i in range(len_bat)]
    bat_rlocalisations = [i for i in range(len_bat)]

    # bat and init-setup:2/2
    for bat_i in range(len_bat):
        bat_rpredictions[bat_i] = [xxxx for xxxx in range(len_rpredictions)]
        bat_rlocalisations[bat_i] = [xxxx for xxxx in range(len_rpredictions)]

    for ri in range(len_rpredictions):

        # 2 x H x W x Boxn x Cls
        px = rpredictions[ri]
        lx = rlocalisations[ri]

        # []_2  ele(np): 1 x H x W x Boxn x Cls
        px_bat_list = np.array_split(px, px.shape[0], axis=0)
        lx_bat_list = np.array_split(lx, lx.shape[0], axis=0)

        # batch_process
        for bat_i in range(len_bat):
            dx1 = px_bat_list[bat_i]
            dx2 = lx_bat_list[bat_i]

            bat_rpredictions[bat_i][ri] = dx1
            bat_rlocalisations[bat_i][ri] = dx2

    return bat_rpredictions,bat_rlocalisations


# Main image processing routine.
def process_image(imgs, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):

    # Run SSD network.

    rpredictions, rlocalisations = isess.run([predictions, localisations], feed_dict={img_input: imgs})

    bat_rpredictions, bat_rlocalisations = re_construct_result(rpredictions,rlocalisations)


    bat_rclasses = []
    bat_rscores = []
    bat_rbboxes = []

    rbbox_img = np.array([0., 0., 1., 1.])
    for bindex in range(batch_size):

        bti_rpredictions = bat_rpredictions[bindex]
        bti_rlocalisations = bat_rlocalisations[bindex]


        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(bti_rpredictions, bti_rlocalisations, ssd_anchors, select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

        # batch_process
        bat_rclasses.append(rclasses)
        bat_rscores.append(rscores)
        bat_rbboxes.append(rbboxes)
    return bat_rclasses, bat_rscores, bat_rbboxes


img = misc.imread('dog.jpg')
img2 = misc.imread('person.jpg')


def prepare_img(img):
    # VGG mean parameters.
    _R_MEAN = 123.
    _G_MEAN = 117.
    _B_MEAN = 104.

    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    print img
    img = img - means

    print img



img = misc.imresize(img,(300,300,3))
img2 = misc.imresize(img2,(300,300,3))

bat_img = [img,img2]

bat_rclasses, bat_rscores, bat_rbboxes = process_image(bat_img)

for i in range(batch_size):
    visualization.plt_bboxes(bat_img[i], bat_rclasses[i], bat_rscores[i], bat_rbboxes[i])
