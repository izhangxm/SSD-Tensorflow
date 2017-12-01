# coding: utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import sys
import MPROCESS

sys.path.append('../')

from nets import ssd_vgg_300 as ssd_vgg_300
from notebooks import visualization
from MTools import Timer

# ------------------------- 设定GPU内存为尽量少用 -------------------------------------------------------------------
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config)

# ------------------------- 计时器 -------------------------------------------------------------------------------
mtime = Timer()

# ------------------------- 参数设定-------------------------------------------------------------------------------
data_format = 'NHWC'
batch_size = 2
net_shape = (300, 300)

# -------------------------------- 网络定义 -----------------------------------------------------------------------
img_4d = tf.to_float(tf.placeholder(tf.uint8, shape=(None, 300, 300, 3)))
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(img_4d, is_training=False, reuse=None)

# ------------------------- 配合tensorboard显示网络结构 ---------------------------------------------------------------
writer = tf.summary.FileWriter("logs/", isess.graph)

# ------------------------- 初始化变量 -----------------------------------------------------------------------------
mtime.start()
isess.run(tf.global_variables_initializer())
mtime.consume('init_varible')

# ------------------------- 恢复模型参数 ----------------------------------------------------------------------------
mtime.start()
# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
mtime.consume('restore_model')

# -------------------------  数据读入并处理 --------------------------------------------------------------------------
mtime.start()
img = misc.imread('dog.jpg')
img2 = misc.imread('person.jpg')

p_img = MPROCESS.preprocess_img(img, shape=(300, 300, 3))
p_img2 = MPROCESS.preprocess_img(img2, shape=(300, 300, 3))
bat_p_img = [p_img, p_img2]
mtime.consume('preprocess_pic')

# ------------------------- 网络运算，得出结果-------------------------------------------------------------------------
mtime.start()
rpredictions, rlocalisations = isess.run([predictions, localisations], feed_dict={img_4d: bat_p_img})
mtime.consume('run_foward')

# ------------------------- 重构结果，方便多进程处理--------------------------------------------------------------------
mtime.start()
bat_rpredictions, bat_rlocalisations = MPROCESS.re_construct_result(rpredictions, rlocalisations)
mtime.consume('reconstuct_result')

# ------------------------- 选择最优结果-----------------------------------------------------------------------------
mtime.start()

# 方法1
# bat_rclasses, bat_rscores, bat_rbboxes = MPROCESS.select_batch_result(bat_rpredictions, bat_rlocalisations, ssd_anchors=ssd_net.anchors(net_shape))

# 方法2
bat_rclasses = []
bat_rscores = []
bat_rbboxes = []
for i in range(len(bat_rpredictions)):
    rclasses, rscores, rbboxes = MPROCESS.select_single_result(bat_rpredictions[i], bat_rlocalisations[i], ssd_anchors=ssd_net.anchors(net_shape))
    bat_rclasses.append(rclasses)
    bat_rscores.append(rscores)
    bat_rbboxes.append(rbboxes)

mtime.consume('find_box')

# ------------------------- 显示 -----------------------------------------------------------------------------------
bat_normal_img = [img, img2]
for i in range(batch_size):
    visualization.plt_bboxes(bat_normal_img[i], bat_rclasses[i], bat_rscores[i], bat_rbboxes[i])


### 这个文件重新整理了逻辑，通用步骤放置在了MPROCESS文件中，主程序更加简洁
