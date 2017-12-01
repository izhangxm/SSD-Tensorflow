# coding: utf-8
from scipy import misc
import numpy as np
from nets import ssd_vgg_300 as ssd_vgg_300, np_methods
"""
通用处理函数
"""

"""

# 此与处理方法与TensorFlow处理后，不同之处在于TensorFlow是小数，本方法只能到整数，但模型运行结果一样
@:param img ? x ? x 3
@:return img shape=shape
"""

def preprocess_img(img, shape=(300, 300, 3)):

    # VGG mean parameters.
    _R_MEAN = 123.
    _G_MEAN = 117.
    _B_MEAN = 104.

    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    img = misc.imresize(img, shape, interp='bilinear')
    return  img - means



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

"""
选择合适的框和分类，处理一批图片结果

"""
def select_batch_result(bat_rpredictions,bat_rlocalisations,ssd_anchors=None,select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    if ssd_anchors is None:
        raise Exception('ssd_anchors is None')


    bat_rclasses = []
    bat_rscores = []
    bat_rbboxes = []

    rbbox_img = np.array([0., 0., 1., 1.])
    batch_size = len(bat_rpredictions)

    for bindex in range(batch_size):
        bti_rpredictions = bat_rpredictions[bindex]
        bti_rlocalisations = bat_rlocalisations[bindex]

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(bti_rpredictions, bti_rlocalisations, ssd_anchors,
                                                                  select_threshold=select_threshold,
                                                                  img_shape=net_shape, num_classes=21, decode=True)

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


"""
选择合适的框和分类，只处理单个img，适合多进程调用
"""
def select_single_result(bti_rpredictions,bti_rlocalisations,ssd_anchors=None,select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    if ssd_anchors is None:
        raise Exception('ssd_anchors is None')

    rbbox_img = np.array([0., 0., 1., 1.])

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(bti_rpredictions, bti_rlocalisations, ssd_anchors,
                                                              select_threshold=select_threshold,
                                                              img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)


    return rclasses, rscores, rbboxes